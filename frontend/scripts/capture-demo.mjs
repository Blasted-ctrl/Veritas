// Real API evidence only: no request interception or fabricated predictions.
import { chromium, expect } from "@playwright/test";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const { values } = parseArgs({ options: {
  url: { type: "string", default: "http://127.0.0.1:3000" },
  output: { type: "string", default: "../.turbo/demo-captures" },
} });
const output = resolve(values.output);
const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || undefined });
const context = await browser.newContext({
  viewport: { width: 1280, height: 900 }, colorScheme: "light", reducedMotion: "reduce",
});
const page = await context.newPage();
page.setDefaultTimeout(180_000);
const errors = [];
page.on("pageerror", error => errors.push(error.message));
page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
const evidence = { captured_at: new Date().toISOString(), url: values.url, browser: browser.version(), analyses: [], screenshots: [] };
try {
  await mkdir(output, { recursive: true });
  await page.goto(values.url);
  await expect(page.getByText("Detector ready", { exact: true })).toBeVisible({ timeout: 30_000 });
  const gif = Buffer.from("R0lGODlhCAAIAIEAAP8AAAAAAAAAAAAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQAZAAAACwAAAAACAAIAAAIDwABCBxIsKDBgwgTKkwYEAAh+QQBZAABACwAAAAACAAIAIEAAP8AAAAAAAAAAAAIDwABCBxIsKDBgwgTKkwYEAAh+QQBZAABACwAAAAACAAIAIEAgAAAAAAAAAAAAAAIDwABCBxIsKDBgwgTKkwYEAAh+QQBZAABACwAAAAACAAIAIH///8AAAAAAAAAAAAIDwABCBxIsKDBgwgTKkwYEAA7", "base64");
  for (const [kind, name] of [["image", "generated-lakehouse.png"], ["video", "generated-demo.mp4"], ["video", "decoder-fixture.gif"]]) {
    const sample = new URL("../public/samples/" + name, import.meta.url);
    await page.locator('input[type="file"]').setInputFiles(name.endsWith(".gif") ? { name, mimeType: "image/gif", buffer: gif } : fileURLToPath(sample));
    const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === "/verify" && response.request().method() === "POST");
    await page.getByRole("button", { name: "Analyze file", exact: true }).click();
    const response = await responsePromise;
    if (!response.ok()) throw new Error("Real inference failed: " + response.status() + " " + await response.text());
    const result = await response.json();
    if (result.modality !== kind || !/^[a-f0-9]{64}$/.test(result.model_info.artifact_sha256)) throw new Error("Missing real model identity");
    await expect(page.getByRole("button", { name: "Analyze again", exact: true })).toBeVisible({ timeout: 180_000 });
    await expect(page.getByRole("meter")).toHaveAttribute("aria-valuenow", String(Number((result.fake_probability * 100).toFixed(1))));
    if (kind === "video") {
      if (!(result.frames_analyzed > 0)) throw new Error("Video has no scored frames");
      const average = result.frames.reduce((sum, frame) => sum + frame.fake_probability, 0) / result.frames.length;
      if (Math.abs(average - result.fake_probability) > 1e-6) throw new Error("Video aggregate differs from frame average");
      await page.locator("summary").filter({ hasText: "View individual frame scores" }).click();
    }
    evidence.analyses.push({ fixture: name, response: result });
    if (name.endsWith(".gif")) continue;
    for (const [width, label] of [[1280, "desktop"], [390, "mobile"]]) {
      await page.setViewportSize({ width, height: 900 });
      for (const theme of ["light", "dark"]) {
        if ((await page.locator("html").getAttribute("data-theme") || "light") !== theme) {
          await page.getByRole("button", { name: "Switch color theme" }).click();
        }
        await expect.poll(async () => await page.locator("html").getAttribute("data-theme") || "light").toBe(theme);
        if (!await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)) throw new Error("Horizontal overflow: " + kind + "/" + label + "/" + theme);
        const filename = kind + "-" + label + "-" + theme + ".jpg";
        await page.screenshot({ path: resolve(output, filename), fullPage: true, type: "jpeg", quality: 88, animations: "disabled" });
        evidence.screenshots.push(filename);
      }
    }
    await page.setViewportSize({ width: 1280, height: 900 });
    if (await page.locator("html").getAttribute("data-theme") !== "light") await page.getByRole("button", { name: "Switch color theme" }).click();
  }
  if (errors.length) throw new Error("Browser errors: " + errors.join("\n"));
  evidence.gif_fixture = "Four 8×8 solid-color frames constructed with Pillow, one second per frame. Decoder/inference plumbing only; no detection-quality claim.";
  evidence.fixture_provenance = JSON.parse(await readFile(new URL("../public/samples/credits.json", import.meta.url), "utf8"));
  await writeFile(resolve(output, "manifest.json"), JSON.stringify(evidence, null, 2) + "\n");
  console.log("Captured real image/video inference in desktop/mobile, light/dark: " + output);
} finally {
  await context.close();
  await browser.close();
}
