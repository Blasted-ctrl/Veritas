// Capture real browser interactions and API responses. No mocked predictions.
import { chromium, expect } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";

const { values } = parseArgs({
  options: {
    url: { type: "string", default: "http://127.0.0.1:3000" },
    output: { type: "string", default: "../.turbo/demo-recording" },
  },
});
const output = resolve(values.output);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
});
const evidence = {
  recorded_at: new Date().toISOString(),
  url: values.url,
  browser: browser.version(),
  note: "Real-time browser recordings, real API responses. Public sample files only.",
  clips: [],
};
try {
  for (const [id, filename] of [
    ["photo", "earthrise.jpg"],
    ["image", "generated-lakehouse.png"],
    ["video", "generated-demo.mp4"],
  ]) {
    const context = await browser.newContext({
      viewport: { width: 1280, height: 900 },
      colorScheme: "light",
      reducedMotion: "reduce",
      recordVideo: { dir: output, size: { width: 1280, height: 900 } },
    });
    try {
      const page = await context.newPage();
      page.setDefaultTimeout(120_000);
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.goto(values.url);
      await expect(
        page.getByText("Detector ready", { exact: true }),
      ).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      const recordingStart = Date.now();
      // Deliberate holds make the demonstration readable at its actual speed.
      await page.waitForTimeout(1000);
      await page
        .locator('input[type="file"]')
        .setInputFiles(
          fileURLToPath(
            new URL("../public/samples/" + filename, import.meta.url),
          ),
        );
      await page.waitForTimeout(1200);
      const responsePromise = page.waitForResponse(
        (response) =>
          new URL(response.url()).pathname === "/verify" &&
          response.request().method() === "POST",
      );
      await page
        .getByRole("button", { name: "Analyze file", exact: true })
        .click();
      const response = await responsePromise;
      if (!response.ok()) throw new Error(await response.text());
      const result = await response.json();
      if (!/^[a-f0-9]{64}$/.test(result.model_info.artifact_sha256))
        throw new Error("No model checksum in response");
      await expect(
        page.getByRole("button", { name: "Analyze again", exact: true }),
      ).toBeVisible();
      await expect(page.getByRole("meter")).toHaveAttribute(
        "aria-valuenow",
        String(Number((result.fake_probability * 100).toFixed(1))),
      );
      await page.waitForTimeout(1700);
      if (id === "video") {
        await page
          .locator("summary")
          .filter({ hasText: "View individual frame scores" })
          .click();
        await page.locator(".frames").scrollIntoViewIfNeeded();
        if (
          result.frames_analyzed !== result.frames.length ||
          result.frames.length === 0
        )
          throw new Error("Invalid frame count");
        const average =
          result.frames.reduce(
            (sum, frame) => sum + frame.fake_probability,
            0,
          ) / result.frames.length;
        if (Math.abs(average - result.fake_probability) > 1e-6)
          throw new Error("Incorrect frame average");
      } else {
        await page
          .locator("summary")
          .filter({ hasText: "Model & analysis details" })
          .click();
      }
      await page.waitForTimeout(4500);
      await page.screenshot({ path: resolve(output, id + "-result.png") });
      if (errors.length) throw new Error(errors.join("\n"));
      const video = page.video();
      const elapsed = (Date.now() - recordingStart) / 1000;
      await context.close();
      await video.saveAs(resolve(output, id + ".webm"));
      evidence.clips.push({
        id,
        fixture: filename,
        seconds_after_ready: elapsed,
        file: id + ".webm",
        response: result,
      });
      console.log(
        id +
          ": " +
          result.verdict +
          " / " +
          (result.fake_probability * 100).toFixed(1) +
          "%",
      );
    } finally {
      await context.close();
    }
  }
  await writeFile(
    resolve(output, "manifest.json"),
    JSON.stringify(evidence, null, 2) + "\n",
  );
} finally {
  await browser.close();
}
