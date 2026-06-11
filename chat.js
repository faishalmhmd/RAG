import ollama from "ollama";
import mysql from "mysql2/promise";

const question = process.argv.slice(2).join(" ");

if (!question) {
  console.log('node chat.js "pertanyaan kamu"');
  process.exit(1);
}

// =========================
// CONFIG
// =========================
const TOP_K = 3;
const MIN_SCORE = 0.3; // ambang batas similarity, dokumen di bawah ini dianggap tidak relevan
const ROW_LIMIT = 5000; // batas jumlah dokumen yang diambil dari DB

// =========================
// SAFE COSINE
// =========================
function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;

  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < a.length; i++) {
    const x = a[i] || 0;
    const y = b[i] || 0;

    dot += x * y;
    magA += x * x;
    magB += y * y;
  }

  const denom = Math.sqrt(magA) * Math.sqrt(magB);

  if (denom === 0) return 0;

  return dot / denom;
}

// =========================
// PARSE EMBEDDING (FIX UTAMA)
// =========================
function parseEmbedding(raw) {
  if (Array.isArray(raw)) return raw;

  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }

  return null;
}

// =========================
// SPINNER (LOADING ANIMATION)
// =========================
function startSpinner(label = "Loading") {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  let i = 0;

  process.stdout.write("\x1B[?25l"); // hide cursor

  const interval = setInterval(() => {
    process.stdout.write(`\r${frames[i]} ${label}`);
    i = (i + 1) % frames.length;
  }, 80);

  return () => {
    clearInterval(interval);
    process.stdout.write(`\r\x1B[K`); // clear line
    process.stdout.write("\x1B[?25h"); // show cursor
  };
}

// =========================
// HELPER: FORMAT BYTES
// =========================
function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${units[i]}`;
}

// =========================
// HELPER: SNAPSHOT RESOURCE USAGE
// =========================
function snapshotUsage() {
  return {
    cpu: process.cpuUsage(), // { user, system } dalam microseconds
    mem: process.memoryUsage(), // rss, heapTotal, heapUsed, external, arrayBuffers
    time: process.hrtime.bigint(),
  };
}

function diffUsage(start, end) {
  const cpuUserMs = (end.cpu.user - start.cpu.user) / 1000;
  const cpuSystemMs = (end.cpu.system - start.cpu.system) / 1000;
  const wallMs = Number(end.time - start.time) / 1e6;

  return {
    cpuUserMs,
    cpuSystemMs,
    cpuTotalMs: cpuUserMs + cpuSystemMs,
    wallMs,
  };
}

async function getOllamaMemoryUsage() {
  try {
    const res = await fetch("http://127.0.0.1:11434/api/ps");
    const data = await res.json();
    return data.models || [];
  } catch (err) {
    return [];
  }
}

let db;
let stopSpinner = () => {};

// Tracking total token usage dari semua call ke ollama
let totalPromptTokens = 0;
let totalCompletionTokens = 0;

// Snapshot resource di awal program
const overallStart = snapshotUsage();

try {
  // =========================
  // DB CONNECT
  // =========================
  db = await mysql.createConnection({
    host: "127.0.0.1",
    port: 4000,
    user: "root",
    database: "rag",
  });

  // =========================
  // QUESTION EMBEDDING
  // =========================
  console.log("\n📥 Question:");
  console.log(question);

  stopSpinner = startSpinner("Embedding pertanyaan...");

  const embedStart = snapshotUsage();
  const embed = await ollama.embeddings({
    model: "bge-m3",
    prompt: question,
  });
  const embedEnd = snapshotUsage();

  stopSpinner();

  // Beberapa versi ollama mengembalikan prompt_eval_count untuk embeddings
  if (typeof embed.prompt_eval_count === "number") {
    totalPromptTokens += embed.prompt_eval_count;
  }

  const embedDiff = diffUsage(embedStart, embedEnd);

  const queryVec = embed.embedding;

  // =========================
  // LOAD DATA
  // =========================
  console.log("\n📡 Loading documents...");

  stopSpinner = startSpinner("Mengambil dokumen dari database...");

  const dbStart = snapshotUsage();
  const [rows] = await db.execute(
    `SELECT content, embedding FROM documents LIMIT ${Number(ROW_LIMIT)}`
  );
  const dbEnd = snapshotUsage();

  stopSpinner();

  const dbDiff = diffUsage(dbStart, dbEnd);

  console.log(`📦 Documents loaded: ${rows.length}`);

  // =========================
  // PARSE + SCORE
  // =========================
  const scoreStart = snapshotUsage();

  const scored = rows
    .map((r) => {
      const vec = parseEmbedding(r.embedding);
      if (!vec) return null;

      return {
        content: r.content,
        score: cosine(queryVec, vec),
      };
    })
    .filter(Boolean);

  const scoreEnd = snapshotUsage();
  const scoreDiff = diffUsage(scoreStart, scoreEnd);

  if (scored.length === 0) {
    console.warn(
      "\n⚠️ Tidak ada embedding valid yang bisa diparse dari database."
    );
  }

  // =========================
  // SORT + FILTER BY THRESHOLD
  // =========================
  scored.sort((a, b) => b.score - a.score);

  const top = scored.slice(0, TOP_K).filter((r) => r.score >= MIN_SCORE);

  console.log("\n🔎 Top scores:");
  if (scored.length > 0) {
    scored
      .slice(0, TOP_K)
      .forEach((t, i) => console.log(`${i + 1}. ${t.score.toFixed(4)}`));
  } else {
    console.log("(tidak ada)");
  }

  if (top.length === 0) {
    console.warn(
      `\n⚠️ Tidak ada dokumen dengan score >= ${MIN_SCORE}. Konteks akan kosong, jawaban mungkin tidak akurat.`
    );
  }

  // =========================
  // BUILD CONTEXT
  // =========================
  const ragContext = top.map((r) => r.content).join("\n\n");

  console.log("\n📄 Context size:", ragContext.length);

  // =========================
  // CHAT LLM
  // =========================
  console.log("\n🤖 Answer:\n");

  const systemPrompt =
    ragContext.length > 0
      ? `Jawab hanya berdasarkan konteks berikut. Jika informasi tidak ada di konteks, katakan tidak tahu.\n\n=== CONTEXT ===\n${ragContext}`
      : `Tidak ada konteks relevan yang ditemukan di database. Beri tahu pengguna bahwa informasi tidak ditemukan, jangan mengarang jawaban.`;

  const chatStart = snapshotUsage();

  const stream = await ollama.chat({
    model: "gemma3",
    stream: true,
    options: {
      num_ctx: 8192, // atau lebih, tergantung VRAM
    },
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: question },
    ],
  });

  let answer = "";
  let firstChunk = true;
  let lastChunk = null;
  stopSpinner = startSpinner("Menyusun jawaban...");

  for await (const chunk of stream) {
    if (firstChunk) {
      stopSpinner();
      firstChunk = false;
    }

    const text = chunk.message?.content || "";
    process.stdout.write(text);
    answer += text;

    lastChunk = chunk; // simpan chunk terakhir (biasanya berisi stats saat done: true)
  }

  if (firstChunk) stopSpinner(); // jaga-jaga kalau stream kosong

  const chatEnd = snapshotUsage();
  const chatDiff = diffUsage(chatStart, chatEnd);

  // Ambil token usage dari chunk terakhir (done: true)
  if (lastChunk) {
    if (typeof lastChunk.prompt_eval_count === "number") {
      totalPromptTokens += lastChunk.prompt_eval_count;
    }
    if (typeof lastChunk.eval_count === "number") {
      totalCompletionTokens += lastChunk.eval_count;
    }
  }

  console.log("\n");

  // =========================
  // FINAL RESOURCE SNAPSHOT
  // =========================
  const overallEnd = snapshotUsage();
  const overallDiff = diffUsage(overallStart, overallEnd);
  const mem = process.memoryUsage();

  // =========================
  // STATS
  // =========================
  // console.log("────────────────────");
  // console.log("📊 STATS");
  // console.log("────────────────────");
  // console.log("Documents loaded:", rows.length);
  // console.log("Valid embeddings:", scored.length);
  // console.log("Top K (after threshold):", top.length);
  // console.log("Answer length:", answer.length);

  // console.log("\n🔢 TOKEN USAGE");
  // console.log("Prompt tokens   :", totalPromptTokens);
  // // console.log("Completion tokens:", totalCompletionTokens);
  // console.log("Total tokens    :", totalPromptTokens + totalCompletionTokens);

  // // console.log("\n⏱️ TIMING (wall time)");
  // // console.log("Embedding   :", embedDiff.wallMs.toFixed(2), "ms");
  // // console.log("DB query    :", dbDiff.wallMs.toFixed(2), "ms");
  // // console.log("Scoring     :", scoreDiff.wallMs.toFixed(2), "ms");
  // // console.log("Chat (LLM)  :", chatDiff.wallMs.toFixed(2), "ms");
  // // console.log("Total       :", overallDiff.wallMs.toFixed(2), "ms");

  // // console.log("\n🧠 CPU USAGE (proses node, total)");
  // // console.log("User CPU time  :", overallDiff.cpuUserMs.toFixed(2), "ms");
  // // console.log("System CPU time:", overallDiff.cpuSystemMs.toFixed(2), "ms");
  // // console.log("Total CPU time :", overallDiff.cpuTotalMs.toFixed(2), "ms");
  // // console.log(
  // //   "CPU usage (%)  :",
  // //   ((overallDiff.cpuTotalMs / overallDiff.wallMs) * 100).toFixed(1) + "%"
  // // );

  // // console.log("\n💾 MEMORY USAGE (akhir proses)");
  // // console.log("RSS         :", formatBytes(mem.rss));
  // // console.log("Heap Total  :", formatBytes(mem.heapTotal));
  // // console.log("Heap Used   :", formatBytes(mem.heapUsed));
  // // console.log("External    :", formatBytes(mem.external));
  // // console.log("Array Buffers:", formatBytes(mem.arrayBuffers));

  // console.log("\n🦙 OLLAMA LOADED MODELS");
  // const ollamaModels = await getOllamaMemoryUsage();
  // if (ollamaModels.length === 0) {
  //   console.log("(tidak ada info / Ollama tidak merespons)");
  // } else {
  //   ollamaModels.forEach((m) => {
  //     console.log(`- ${m.name}`);
  //     console.log(`  Total size : ${formatBytes(m.size)}`);
  //     console.log(`  VRAM size  : ${formatBytes(m.size_vram)}`);
  //     console.log(`  Expires at : ${m.expires_at}`);
  //   });
  // }
} catch (err) {
  stopSpinner();
  console.error("\n❌ Error:", err.message);
  process.exitCode = 1;
} finally {
  if (db) await db.end();
}