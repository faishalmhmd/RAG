import fs from "fs";
import readline from "readline";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import mysql from "mysql2/promise";
import { execSync } from "child_process";
import { createWorker } from "tesseract.js";

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
  return data.embedding;
}

// =====================
// INPUT LEVEL
// =====================
console.log("\n========================");
console.log("RAG INGEST MENU");
console.log("========================");
console.log("1 = Text only (fast)");
console.log("2 = Text + OCR untuk halaman yang text-layer-nya minim (recommended)");
console.log("3 = Text + OCR untuk SEMUA halaman (slow, paling akurat utk tabel berat)");
console.log("========================\n");

const levelInput = await ask("Pilih level (1/2/3): ");
const LEVEL = Number(levelInput);

console.log("\n🚀 Running LEVEL:", LEVEL);

// =====================
// LOAD PDF
// =====================
const filePath = "./files/LBP-PPATK-Semester-I-2025.pdf";

const pdf = await pdfjsLib.getDocument({
  data: new Uint8Array(fs.readFileSync(filePath)),
  standardFontDataUrl: "node_modules/pdfjs-dist/standard_fonts/",
}).promise;

console.log("📄 Pages:", pdf.numPages);

// =====================
// TEXT EXTRACTION (POSITIONAL / LAYOUT-AWARE)
// =====================
// Rekonstruksi baris berdasarkan posisi Y agar urutan teks (terutama
// angka di tabel) tidak acak akibat urutan render PDF yang tidak linear.
function extractPageText(content) {
  // Urutkan item berdasarkan Y (atas ke bawah), lalu X (kiri ke kanan)
  const items = content.items
    .filter((it) => it.str !== undefined)
    .map((it) => ({
      str: it.str,
      x: it.transform[4],
      y: it.transform[5],
    }))
    .sort((a, b) => {
      // Y di PDF dihitung dari bawah, jadi urutkan descending utk top->bottom
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
      // baris baru
      text += "\n" + item.str;
    } else {
      // baris sama, cek jarak X untuk menentukan perlu spasi atau tidak
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
  // fallback kasar kalau item.width tidak tersedia
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

// =====================
// CHUNK (LINE/PARAGRAPH AWARE + OVERLAP)
// =====================
// Memecah berdasarkan baris, bukan potong karakter mentah, supaya
// angka/kalimat tidak terpotong di tengah saat melewati batas size.
function chunk(text, size = 1000, overlap = 100) {
  const lines = text.split("\n");
  const chunks = [];
  let current = "";

  for (const line of lines) {
    if (current.length + line.length + 1 > size && current.length > 0) {
      chunks.push(current.trim());

      // overlap: ambil ekor chunk sebelumnya supaya konteks nyambung
      const tail = current.slice(-overlap);
      current = tail + "\n" + line + "\n";
    } else {
      current += line + "\n";
    }
  }

  if (current.trim()) chunks.push(current.trim());

  return chunks.filter((c) => c.length > 0);
}

const chunks = chunk(rawText);

// =====================
// OCR
// =====================
function renderImages() {
  console.log("🖼️ Rendering images (300 DPI)...");

  execSync(`rm -rf ./tmp && mkdir -p ./tmp`);
  // -r 300 -> resolusi lebih tinggi, penting agar Tesseract bisa baca
  // angka kecil di tabel dengan benar
  execSync(`pdftoppm -png -r 300 "${filePath}" ./tmp/page`);

  return fs
    .readdirSync("./tmp")
    .filter((f) => f.endsWith(".png"))
    .sort() // pastikan urutan halaman benar (page-1, page-2, ...)
    .map((f) => `./tmp/${f}`);
}

// PSM 6 = "Assume a single uniform block of text", umumnya lebih stabil
// untuk halaman tabel/laporan dibanding mode auto default.
async function ocrImages(images, pageIndices) {
  console.log("🔍 OCR running (ini bisa lama)...");

  const worker = await createWorker("eng");
  await worker.setParameters({
    tessedit_pageseg_mode: "6",
    // izinkan karakter umum laporan keuangan: angka, titik, koma, %, Rp, dll
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

// Tentukan halaman mana saja yang perlu di-OCR
function pagesNeedingOcr() {
  if (LEVEL === 3) {
    // semua halaman
    return pageTextLengths.map((_, i) => i + 1);
  }
  if (LEVEL === 2) {
    // hanya halaman yang text-layer-nya sangat minim
    // (kemungkinan halaman scan / gambar / tabel kompleks tanpa teks asli)
    const THRESHOLD = 30; // jumlah karakter non-spasi minimal dianggap "ada teks"
    return pageTextLengths
      .map((len, i) => (len < THRESHOLD ? i + 1 : null))
      .filter((p) => p !== null);
  }
  return [];
}

// =====================
// INSERT
// =====================
async function insert(content, type, meta = {}) {
  if (!content || !content.trim()) return;

  const emb = await embed(content);

  await db.execute(
    `INSERT INTO documents (content, type, embedding, meta) VALUES (?, ?, ?, ?)`,
    [content, type, JSON.stringify(emb), JSON.stringify(meta)]
  );
}

// =====================
// RUN PIPELINE
// =====================
for (let i = 0; i < chunks.length; i++) {
  await insert(chunks[i], "text", { chunkIndex: i });
  console.log(`[TEXT ${i + 1}/${chunks.length}]`);
}

if (LEVEL >= 2) {
  const targetPages = pagesNeedingOcr();

  if (targetPages.length === 0) {
    console.log("ℹ️ Tidak ada halaman yang perlu OCR (text-layer sudah cukup).");
  } else {
    console.log(`ℹ️ Halaman yang akan di-OCR: ${targetPages.join(", ")}`);

    const allImages = renderImages();

    // pdftoppm menamai file page-1.png, page-2.png, dst (atau dengan padding,
    // tergantung versi poppler). Map nomor halaman -> file image.
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
      await insert(text, "image", { page });
      console.log(`[OCR page ${page}] (${i + 1}/${ocrResults.length})`);
    }
  }
}

console.log("\n✅ DONE");

rl.close();
await db.end();