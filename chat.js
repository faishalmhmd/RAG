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
// mysql2 otomatis parse kolom JSON jadi array/object JS,
// jadi tidak boleh di-JSON.parse() lagi kalau sudah berupa array.
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

let db;
let stopSpinner = () => {};

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

  const embed = await ollama.embeddings({
    model: "bge-m3",
    prompt: question,
  });

  stopSpinner();

  const queryVec = embed.embedding;

  // =========================
  // LOAD DATA
  // =========================
  console.log("\n📡 Loading documents...");

  stopSpinner = startSpinner("Mengambil dokumen dari database...");

  const [rows] = await db.execute(
    `SELECT content, embedding FROM documents LIMIT ${Number(ROW_LIMIT)}`
  );

  stopSpinner();

  console.log(`📦 Documents loaded: ${rows.length}`);

  // =========================
  // PARSE + SCORE
  // =========================
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

  const stream = await ollama.chat({
    model: "gemma3",
    stream: true,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: question },
    ],
  });

  let answer = "";
  let firstChunk = true;
  stopSpinner = startSpinner("Menyusun jawaban...");

  for await (const chunk of stream) {
    if (firstChunk) {
      stopSpinner();
      firstChunk = false;
    }

    const text = chunk.message?.content || "";
    process.stdout.write(text);
    answer += text;
  }

  if (firstChunk) stopSpinner(); // jaga-jaga kalau stream kosong

  console.log("\n");

  // =========================
  // STATS
  // =========================
  console.log("────────────────────");
  console.log("📊 STATS");
  console.log("────────────────────");
  console.log("Documents loaded:", rows.length);
  console.log("Valid embeddings:", scored.length);
  console.log("Top K (after threshold):", top.length);
  console.log("Answer length:", answer.length);
} catch (err) {
  stopSpinner();
  console.error("\n❌ Error:", err.message);
  process.exitCode = 1;
} finally {
  if (db) await db.end();
}