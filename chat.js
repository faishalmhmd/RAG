import ollama from "ollama";
import mysql from "mysql2/promise";
import os from "os";

const question = process.argv.slice(2).join(" ");

if (!question) {
  console.log("Usage:");
  console.log('node chat.js "pertanyaan kamu"');
  process.exit(1);
}

let spinner;

function startLoading(text) {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

  let i = 0;

  spinner = setInterval(() => {
    process.stdout.write(
      `\r${frames[i++ % frames.length]} ${text}`
    );
  }, 100);
}

function stopLoading(text = "") {
  clearInterval(spinner);
  process.stdout.write(`\r✓ ${text}\n`);
}

console.time("Total Time");

const db = await mysql.createConnection({
  host: "127.0.0.1",
  port: 4000,
  user: "root",
  database: "rag",
});

try {
  console.log("\n📥 Question:");
  console.log(question);
  console.log("");

  // ==========================================
  // Generate Embedding
  // ==========================================
  startLoading("Generating embedding...");

  const embed = await ollama.embed({
    model: "bge-m3",
    input: question,
  });

  stopLoading("Embedding generated");

  // ==========================================
  // Search Vector
  // ==========================================
  startLoading("Searching TiDB...");

  const [rows] = await db.execute(
    `
    SELECT content
    FROM documents
    ORDER BY VEC_COSINE_DISTANCE(
      embedding,
      ?
    )
    LIMIT 3
    `,
    [JSON.stringify(embed.embeddings[0])]
  );

  stopLoading(`Found ${rows.length} chunks`);

  const ragContext = rows
    .map((r) => r.content)
    .join("\n\n");

  console.log(
    `📄 Context Size: ${ragContext.length} chars`
  );

  const systemInfo = `
Hostname: ${os.hostname()}
Platform: ${os.platform()}
CPU Cores: ${os.cpus().length}
Free Memory: ${(os.freemem() / 1024 / 1024 / 1024).toFixed(2)} GB
Total Memory: ${(os.totalmem() / 1024 / 1024 / 1024).toFixed(2)} GB
Load Average: ${os.loadavg().join(", ")}
Uptime: ${(os.uptime() / 3600).toFixed(2)} jam
`;

  
  console.log("\n🤖 Generating answer...\n");

  const stream = await ollama.chat({
    model: "gemma3",
    stream: true,
    messages: [
      {
        role: "system",
        content: `
Jawab hanya berdasarkan informasi berikut.

=== RAG Context ===
${ragContext}

=== System Info ===
${systemInfo}
`,
      },
      {
        role: "user",
        content: question,
      },
    ],
  });

  let answer = "";
  let tokenCount = 0;

  for await (const chunk of stream) {
    const content = chunk.message?.content || "";

    process.stdout.write(content);

    answer += content;

    if (content.trim()) {
      tokenCount++;
    }
  }

  console.log("\n");

  console.log("────────────────────────────");
  console.log("📊 Stats");
  console.log("────────────────────────────");
  console.log(`Chunks Retrieved : ${rows.length}`);
  console.log(`Answer Length    : ${answer.length} chars`);
  console.log(`Stream Chunks    : ${tokenCount}`);

  console.timeEnd("Total Time");
} catch (err) {
  console.error("\n❌ Error");
  console.error(err);
} finally {
  await db.end();
}