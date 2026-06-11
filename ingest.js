import fs from "fs";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import mysql from "mysql2/promise";
import { execSync } from "child_process";
import Tesseract from "tesseract.js";

// =====================
// DB CONNECT
// =====================
const db = await mysql.createConnection({
  host: "127.0.0.1",
  port: 4000,
  user: "root",
  database: "rag",
});

// =====================
// OLLAMA EMBEDDING (HTTP - STABLE)
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
// LOAD PDF
// =====================
const filePath = "./files/LBP-PPATK-Semester-I-2025.pdf";
const data = new Uint8Array(fs.readFileSync(filePath));

const pdf = await pdfjsLib.getDocument({
  data,
  standardFontDataUrl: "node_modules/pdfjs-dist/standard_fonts/",
}).promise;

console.log("📄 PDF loaded:", pdf.numPages);

// =====================
// TEXT EXTRACTION
// =====================
let rawText = "";

for (let i = 1; i <= pdf.numPages; i++) {
  const page = await pdf.getPage(i);
  const content = await page.getTextContent();

  const pageText = content.items.map((t) => t.str).join(" ");
  rawText += "\n" + pageText;
}

console.log("🧠 Text extracted");

// =====================
// CHUNKING
// =====================
function chunkText(text, size = 1000) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

const textChunks = chunkText(rawText);

console.log("Chunks:", textChunks.length);

// =====================
// IMAGE + OCR (POPPLER SAFE)
// =====================
function renderPdfToImages(file) {
  console.log("🖼️ Rendering PDF to images...");

  execSync(`rm -rf ./tmp && mkdir -p ./tmp`);
  execSync(`pdftoppm -png "${file}" ./tmp/page`);

  const files = fs
    .readdirSync("./tmp")
    .filter((f) => f.endsWith(".png"))
    .map((f) => `./tmp/${f}`);

  return files;
}

const images = renderPdfToImages(filePath);

console.log("Images:", images.length);

// =====================
// OCR IMAGES
// =====================
async function ocrImages(files) {
  const results = [];

  for (const file of files) {
    const ocr = await Tesseract.recognize(file, "eng");

    if (ocr.data.text.trim().length > 0) {
      results.push({
        file,
        text: ocr.data.text,
      });
    }
  }

  return results;
}

const imageChunks = await ocrImages(images);

console.log("OCR done:", imageChunks.length);

// =====================
// INSERT TEXT CHUNKS
// =====================
for (let i = 0; i < textChunks.length; i++) {
  const chunk = textChunks[i];

  const embedding = await embed(chunk);

  await db.execute(
    `INSERT INTO documents (content, type, embedding) VALUES (?, ?, ?)`,
    [chunk, "text", JSON.stringify(embedding)]
  );

  console.log(`[TEXT ${i + 1}] inserted`);
}

// =====================
// INSERT IMAGE CHUNKS
// =====================
for (let i = 0; i < imageChunks.length; i++) {
  const item = imageChunks[i];

  const embedding = await embed(item.text);

  await db.execute(
    `INSERT INTO documents (content, type, embedding) VALUES (?, ?, ?)`,
    [item.text, "image", JSON.stringify(embedding)]
  );

  console.log(`[IMAGE ${i + 1}] inserted`);
}

console.log("✅ DONE RAG PIPELINE");