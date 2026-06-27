// One-command local setup. Cross-platform (Windows/macOS/Linux).
//   npm run setup      → prepares .env, starts the DB (if Docker), migrates, seeds
//   npm run dev        → opens the app at http://localhost:3000
//
// Safe to re-run: it never overwrites an existing .env and skips steps already done.

import { execSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync, copyFileSync } from "node:fs";
import crypto from "node:crypto";

const log = (m) => console.log(`\x1b[36m▶\x1b[0m ${m}`);
const ok = (m) => console.log(`\x1b[32m✓\x1b[0m ${m}`);
const warn = (m) => console.log(`\x1b[33m!\x1b[0m ${m}`);

function has(cmd) {
  const probe = process.platform === "win32" ? "where" : "which";
  return spawnSync(probe, [cmd], { stdio: "ignore" }).status === 0;
}

// 1) .env ---------------------------------------------------------------------
if (!existsSync(".env")) {
  copyFileSync(".env.example", ".env");
  let env = readFileSync(".env", "utf8");
  const key = () => crypto.randomBytes(32).toString("base64");
  env = env
    .replace(/^APP_ENCRYPTION_KEY=.*$/m, `APP_ENCRYPTION_KEY="${key()}"`)
    .replace(/^SESSION_SECRET=.*$/m, `SESSION_SECRET="${key()}"`);
  writeFileSync(".env", env);
  ok("Created .env with freshly generated encryption + session keys.");
} else {
  ok(".env already exists — leaving it untouched.");
}

// 2) Database -----------------------------------------------------------------
const dockerUp =
  has("docker") &&
  spawnSync("docker", ["info"], { stdio: "ignore" }).status === 0;

if (dockerUp) {
  log("Starting Postgres via Docker…");
  try {
    execSync("docker compose up -d", { stdio: "inherit" });
    // Wait for the container healthcheck to go green.
    log("Waiting for the database to be ready…");
    for (let i = 0; i < 30; i++) {
      const id = execSync("docker compose ps -q db").toString().trim();
      const health = id
        ? execSync(`docker inspect --format "{{.State.Health.Status}}" ${id}`)
            .toString()
            .trim()
        : "";
      if (health === "healthy") break;
      await new Promise((r) => setTimeout(r, 2000));
    }
    ok("Database is up.");
  } catch {
    warn("Could not start Docker Postgres. Start a database manually and set DATABASE_URL in .env.");
  }
} else {
  warn(
    "Docker not detected. Start Postgres yourself (or `docker compose up -d`) and ensure\n" +
      "  DATABASE_URL in .env points at it, then re-run `npm run setup`."
  );
}

// 3) Migrate + seed -----------------------------------------------------------
try {
  log("Applying database schema…");
  execSync("npx prisma migrate deploy", { stdio: "inherit" });
  execSync("npx prisma generate", { stdio: "inherit" });
  ok("Schema applied and Prisma client generated.");
} catch {
  warn("Migration failed — is the database running and DATABASE_URL correct?");
  process.exit(1);
}

try {
  log("Seeding a demo invite…");
  execSync("npm run db:seed", { stdio: "inherit" });
} catch {
  warn("Seed skipped (not critical).");
}

console.log("\n\x1b[32mKlart!\x1b[0m Kör nu:  \x1b[1mnpm run dev\x1b[0m");
console.log("Öppna sedan \x1b[1mhttp://localhost:3000\x1b[0m i webbläsaren.\n");
