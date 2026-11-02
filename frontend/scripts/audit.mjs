import { chromium } from "@playwright/test";
import lighthouse from "lighthouse";
import desktopConfig from "lighthouse/core/config/desktop-config.js";
import { createServer } from "node:net";
import { parseArgs } from "node:util";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const { values } = parseArgs({ options: {
  url: { type: "string", default: "http://127.0.0.1:3000" },
  output: { type: "string", default: "../.turbo/lighthouse" },
} });
const output = resolve(values.output);
await mkdir(output, { recursive: true });
// Let the OS choose an available local debugging port for this isolated browser.
const reservation = createServer();
await new Promise((resolve, reject) => { reservation.once("error", reject); reservation.listen(0, "127.0.0.1", resolve); });
const port = reservation.address().port;
await new Promise((resolve, reject) => reservation.close(error => error ? reject(error) : resolve()));
const browser = await chromium.launch({
  channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
  executablePath: process.env.CHROME_PATH || undefined,
  args: ["--remote-debugging-port=" + port, "--remote-debugging-address=127.0.0.1"],
});
const reports = [];
try {
for (const mode of ["mobile", "desktop"]) {
  const path = resolve(output, mode + ".json");
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error("Lighthouse exceeded its five-minute deadline")), 300_000);
  });
  let result;
  try {
    result = await Promise.race([
      lighthouse(values.url, { port, output: "json", onlyCategories: ["performance", "accessibility", "best-practices", "seo"] }, mode === "desktop" ? desktopConfig : undefined),
      deadline,
    ]);
  } finally { clearTimeout(timer); }
  if (!result) throw new Error("Lighthouse produced no report");
  const report = result.lhr;
  await writeFile(path, result.report);
  if (report.runtimeError) throw new Error(JSON.stringify(report.runtimeError));
  reports.push({
    mode, fetched_at: report.fetchTime, lighthouse_version: report.lighthouseVersion,
    url: report.finalDisplayedUrl, environment: report.environment,
    settings: { formFactor: report.configSettings.formFactor, throttling: report.configSettings.throttling },
    scores: Object.fromEntries(Object.entries(report.categories).map(([id, category]) => [id, Math.round(category.score * 100)])),
    metrics: Object.fromEntries(["first-contentful-paint", "largest-contentful-paint", "total-blocking-time", "cumulative-layout-shift"].map(id => [id, report.audits[id].numericValue])),
    warnings: report.runWarnings,
    failed_audits: Object.values(report.audits).filter(audit => audit.score !== null && audit.score < 1 && audit.scoreDisplayMode !== "informative").map(audit => ({ id: audit.id, title: audit.title, score: audit.score, displayValue: audit.displayValue })),
  });
}
} finally { await browser.close(); }
await writeFile(resolve(output, "summary.json"), JSON.stringify({ note: "One local production-build run per mode. Lab scores vary by machine and do not measure detector accuracy.", reports }, null, 2) + "\n");
console.log(JSON.stringify(reports.map(({ mode, scores, metrics }) => ({ mode, scores, metrics })), null, 2));
