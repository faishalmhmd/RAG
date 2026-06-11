  import fs from "fs";
  import path from "path";
  import readline from "readline";
  import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
  import mysql from "mysql2/promise";
  import { execSync } from "child_process";
  import { createWorker } from "tesseract.js";
  import * as XLSX from "xlsx";

  // =====================
  // CLI MENU
  // =====================
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  function ask(q) {
    return new Promise((res) => rl.question(q, res));
  }

  // =====================
  // DB
  // =====================
  const db = await mysql.createConnection({
    host: "127.0.0.1",
    port: 4000,
    user: "root",
    database: "rag",
  });

  // =====================
  // EMBED
  // =====================
  async function embed(text) {
    const res = await fetch("http://localhost:11434/api/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "bge-m3",
        prompt: text,
        
      }),
    });

    const data = await res.json();

    if (!res.ok || !data.embedding) {
      console.error("\n❌ Embedding gagal. Response dari Ollama:");
      console.error(JSON.stringify(data, null, 2));
      console.error("Text yang dikirim (awal 200 char):", text.slice(0, 200));
      throw new Error(data.error || "Embedding response tidak valid (tidak ada field 'embedding')");
    }

    return data.embedding;
  }

  // =====================
  // SCAN FOLDER ./files
  // =====================
  const FILES_DIR = "./files";
  const SUPPORTED_EXT = [".pdf", ".csv", ".xlsx", ".xls"];

  const availableFiles = fs
    .readdirSync(FILES_DIR)
    .filter((f) => SUPPORTED_EXT.includes(path.extname(f).toLowerCase()))
    .sort();

  if (availableFiles.length === 0) {
    console.log(`❌ Tidak ada file yang didukung di folder ${FILES_DIR}`);
    console.log(`   (didukung: ${SUPPORTED_EXT.join(", ")})`);
    rl.close();
    await db.end();
    process.exit(1);
  }

  console.log("\n========================");
  console.log("PILIH FILE");
  console.log("========================");
  availableFiles.forEach((f, i) => {
    console.log(`${i + 1} = ${f}`);
  });
  console.log("========================\n");

  const fileIndexInput = await ask(`Pilih file (1-${availableFiles.length}): `);
  const fileIndex = Number(fileIndexInput) - 1;

  if (
    isNaN(fileIndex) ||
    fileIndex < 0 ||
    fileIndex >= availableFiles.length
  ) {
    console.log("❌ Pilihan tidak valid.");
    rl.close();
    await db.end();
    process.exit(1);
  }

  const selectedFile = availableFiles[fileIndex];
  const filePath = path.join(FILES_DIR, selectedFile);
  const ext = path.extname(selectedFile).toLowerCase();

  console.log(`\n📂 File dipilih: ${selectedFile}`);

  // =====================
  // LEVEL MENU (HANYA UNTUK PDF)
  // =====================
  let LEVEL = 1;

  if (ext === ".pdf") {
    console.log("\n========================");
    console.log("RAG INGEST MENU (PDF)");
    console.log("========================");
    console.log("1 = Text only (fast)");
    console.log("2 = Text + OCR untuk halaman yang text-layer-nya minim (recommended)");
    console.log("3 = Text + OCR untuk SEMUA halaman (slow, paling akurat utk tabel berat)");
    console.log("========================\n");

    const levelInput = await ask("Pilih level (1/2/3): ");
    LEVEL = Number(levelInput);

    console.log("\n🚀 Running LEVEL:", LEVEL);
  }

  // =====================
  // CHUNK (LINE/PARAGRAPH AWARE + OVERLAP)
  // =====================
  function chunk(text, size = 1000, overlap = 100) {
    const lines = text.split("\n");
    const chunks = [];
    let current = "";

    for (const line of lines) {
      if (current.length + line.length + 1 > size && current.length > 0) {
        chunks.push(current.trim());

        const tail = current.slice(-overlap);
        current = tail + "\n" + line + "\n";
      } else {
        current += line + "\n";
      }
    }

    if (current.trim()) chunks.push(current.trim());

    return chunks.filter((c) => c.length > 0);
  }

  // =====================
  // INSERT
  // =====================
  async function insert(content, type, meta = {}) {
    if (!content || !content.trim()) return;

    let emb;
    try {
      emb = await embed(content);
    } catch (err) {
      if (String(err.message).includes("context length") && content.length > 200) {
        // split jadi 2 bagian dan insert terpisah
        console.warn(`⚠️ Chunk terlalu besar (${content.length} chars), membelah jadi 2...`);

        const lines = content.split("\n");
        const mid = Math.ceil(lines.length / 2);
        const part1 = lines.slice(0, mid).join("\n");
        const part2 = lines.slice(mid).join("\n");

        await insert(part1, type, { ...meta, split: "1/2" });
        await insert(part2, type, { ...meta, split: "2/2" });
        return;
      }
      throw err;
    }

    if (!Array.isArray(emb)) {
      throw new Error("Embedding bukan array, tidak bisa disimpan ke DB.");
    }

    await db.execute(
      `INSERT INTO documents (content, type, embedding, meta) VALUES (?, ?, ?, ?)`,
      [content, type, JSON.stringify(emb), JSON.stringify(meta)]
    );
  }

  // =====================================================
  // HANDLER: PDF
  // =====================================================
  async function handlePdf() {
    const pdf = await pdfjsLib.getDocument({
      data: new Uint8Array(fs.readFileSync(filePath)),
      standardFontDataUrl: "node_modules/pdfjs-dist/standard_fonts/",
    }).promise;

    console.log("📄 Pages:", pdf.numPages);

    function extractPageText(content) {
      const items = content.items
        .filter((it) => it.str !== undefined)
        .map((it) => ({
          str: it.str,
          x: it.transform[4],
          y: it.transform[5],
        }))
        .sort((a, b) => {
          if (Math.abs(a.y - b.y) > 2) return b.y - a.y;
          return a.x - b.x;
        });

      let text = "";
      let lastY = null;
      let lastX = null;
      let lastItem = null;

      for (const item of items) {
        if (lastY === null) {
          text += item.str;
        } else if (Math.abs(item.y - lastY) > 2) {
          text += "\n" + item.str;
        } else {
          const gap = item.x - lastX;
          const lastWidth = lastItem.width || 0;
          if (gap > lastWidth * 0.3 + 1) {
            text += " " + item.str;
          } else {
            text += item.str;
          }
        }

        lastY = item.y;
        lastX = item.x + (item.width || estimateWidth(item.str));
        lastItem = { ...item, width: item.width || estimateWidth(item.str) };
      }

      return text;
    }

    function estimateWidth(str) {
      return str.length * 4;
    }

    let rawText = "";
    const pageTextLengths = [];

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      const pageText = extractPageText(content);

      pageTextLengths.push(pageText.replace(/\s+/g, "").length);
      rawText += `\n----- PAGE ${i} -----\n` + pageText + "\n";
    }

    console.log("🧠 Text extracted");

    const chunks = chunk(rawText);

    for (let i = 0; i < chunks.length; i++) {
      await insert(chunks[i], "text", { source: selectedFile, chunkIndex: i });
      console.log(`[TEXT ${i + 1}/${chunks.length}]`);
    }

    // =====================
    // OCR
    // =====================
    function renderImages() {
      console.log("🖼️ Rendering images (300 DPI)...");

      execSync(`rm -rf ./tmp && mkdir -p ./tmp`);
      execSync(`pdftoppm -png -r 300 "${filePath}" ./tmp/page`);

      return fs
        .readdirSync("./tmp")
        .filter((f) => f.endsWith(".png"))
        .sort()
        .map((f) => `./tmp/${f}`);
    }

    async function ocrImages(images, pageIndices) {
      console.log("🔍 OCR running (ini bisa lama)...");

      const worker = await createWorker("eng");
      await worker.setParameters({
        tessedit_pageseg_mode: "6",
        tessedit_char_whitelist:
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,%()/-:Rp ",
      });

      const res = [];

      for (let idx = 0; idx < images.length; idx++) {
        const img = images[idx];
        const pageNo = pageIndices ? pageIndices[idx] : idx + 1;

        const {
          data: { text },
        } = await worker.recognize(img);

        const trimmed = text.trim();
        if (trimmed) {
          res.push({ page: pageNo, text: trimmed });
        }
      }

      await worker.terminate();
      return res;
    }

    function pagesNeedingOcr() {
      if (LEVEL === 3) {
        return pageTextLengths.map((_, i) => i + 1);
      }
      if (LEVEL === 2) {
        const THRESHOLD = 30;
        return pageTextLengths
          .map((len, i) => (len < THRESHOLD ? i + 1 : null))
          .filter((p) => p !== null);
      }
      return [];
    }

    if (LEVEL >= 2) {
      const targetPages = pagesNeedingOcr();

      if (targetPages.length === 0) {
        console.log("ℹ️ Tidak ada halaman yang perlu OCR (text-layer sudah cukup).");
      } else {
        console.log(`ℹ️ Halaman yang akan di-OCR: ${targetPages.join(", ")}`);

        const allImages = renderImages();

        const imageForPage = (pageNo) =>
          allImages.find((f) => {
            const m = f.match(/page-?0*(\d+)\.png$/);
            return m && Number(m[1]) === pageNo;
          });

        const selectedImages = [];
        const selectedPages = [];

        for (const p of targetPages) {
          const img = imageForPage(p);
          if (img) {
            selectedImages.push(img);
            selectedPages.push(p);
          } else {
            console.warn(`⚠️ Gambar untuk halaman ${p} tidak ditemukan, dilewati.`);
          }
        }

        const ocrResults = await ocrImages(selectedImages, selectedPages);

        for (let i = 0; i < ocrResults.length; i++) {
          const { page, text } = ocrResults[i];
          await insert(text, "image", { source: selectedFile, page });
          console.log(`[OCR page ${page}] (${i + 1}/${ocrResults.length})`);
        }
      }
    }
  }

  // =====================================================
  // HANDLER: CSV
  // =====================================================
  async function handleCsv() {
    const raw = fs.readFileSync(filePath, "utf-8");

    // Pisah per baris, lalu chunk seperti teks biasa.
    // Header (baris pertama) diulang di tiap chunk supaya konteks kolom
    // tetap jelas walau chunk-nya berisi baris data di tengah file.
    const lines = raw.split("\n").filter((l) => l.trim().length > 0);

    if (lines.length === 0) {
      console.log("⚠️ File CSV kosong.");
      return;
    }

    const header = lines[0];
    const dataLines = lines.slice(1);

    const ROWS_PER_CHUNK = 30; // sesuaikan tergantung lebar kolom

    const chunks = [];
    for (let i = 0; i < dataLines.length; i += ROWS_PER_CHUNK) {
      const part = dataLines.slice(i, i + ROWS_PER_CHUNK);
      chunks.push(`${header}\n${part.join("\n")}`);
    }

    console.log(`🧠 Total baris data: ${dataLines.length}`);
    console.log(`📦 Total chunk: ${chunks.length}`);

    for (let i = 0; i < chunks.length; i++) {
      await insert(chunks[i], "csv", { source: selectedFile, chunkIndex: i });
      console.log(`[CSV ${i + 1}/${chunks.length}]`);
    }
  }

  // =====================================================
  // HANDLER: XLSX / XLS
  // =====================================================
  function csvEscape(value) {
    const str = String(value ?? "");
    if (str.includes(",") || str.includes('"') || str.includes("\n")) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  }

  async function handleXlsx() {
    const buffer = fs.readFileSync(filePath);
    const workbook = XLSX.read(buffer, { type: "buffer" });

    console.log(`📑 Sheets: ${workbook.SheetNames.join(", ")}`);

    const ROWS_PER_CHUNK = 30;

    for (const sheetName of workbook.SheetNames) {
      const sheet = workbook.Sheets[sheetName];

      // Ambil sebagai array of arrays, biar gampang deteksi baris header
      const rows = XLSX.utils.sheet_to_json(sheet, {
        header: 1,
        defval: "",
        raw: false,
      });

      if (rows.length === 0) {
        console.log(`⚠️ Sheet "${sheetName}" kosong, dilewati.`);
        continue;
      }

      // Cari baris header: baris dengan jumlah sel terisi TERBANYAK
      // di antara beberapa baris pertama. Baris judul/metadata biasanya
      // cuma 1 sel terisi (merged cell), sedangkan header tabel terisi
      // di hampir semua kolom.
      const SCAN_ROWS = Math.min(15, rows.length);
      let headerRowIndex = 0;
      let maxFilled = -1;

      for (let i = 0; i < SCAN_ROWS; i++) {
        const filled = rows[i].filter((c) => String(c).trim() !== "").length;
        if (filled > maxFilled) {
          maxFilled = filled;
          headerRowIndex = i;
        }
      }

      // Baris-baris di atas header dianggap judul/metadata laporan,
      // dimasukkan sebagai konteks tambahan di tiap chunk
      const titleLines = rows
        .slice(0, headerRowIndex)
        .map((r) => r.filter((c) => String(c).trim() !== "").join(" "))
        .filter((l) => l.trim() !== "");

      const titleBlock =
        titleLines.length > 0 ? `${titleLines.join("\n")}\n` : "";

      const header = rows[headerRowIndex].map((c) => String(c).trim()).join(",");

      const dataRows = rows
        .slice(headerRowIndex + 1)
        .filter((r) => r.some((c) => String(c).trim() !== ""))
        .map((r) => r.map((c) => csvEscape(c)).join(","));

      const chunks = [];
      if (dataRows.length === 0) {
        chunks.push(`Sheet: ${sheetName}\n${titleBlock}${header}`);
      } else {
        for (let i = 0; i < dataRows.length; i += ROWS_PER_CHUNK) {
          const part = dataRows.slice(i, i + ROWS_PER_CHUNK);
          chunks.push(
            `Sheet: ${sheetName}\n${titleBlock}${header}\n${part.join("\n")}`
          );
        }
      }

      console.log(
        `🧠 Sheet "${sheetName}" -> header di baris ${headerRowIndex + 1}, ${dataRows.length} baris data, ${chunks.length} chunk`
      );

      for (let i = 0; i < chunks.length; i++) {
        await insert(chunks[i], "xlsx", {
          source: selectedFile,
          sheet: sheetName,
          chunkIndex: i,
        });
        console.log(`[XLSX "${sheetName}" ${i + 1}/${chunks.length}]`);
      }
    }
  }
  // =====================================================
  // RUN PIPELINE
  // =====================================================
  if (ext === ".pdf") {
    await handlePdf();
  } else if (ext === ".csv") {
    await handleCsv();
  } else if (ext === ".xlsx" || ext === ".xls") {
    await handleXlsx();
  }

  console.log("\n✅ DONE");

  rl.close();
  await db.end();