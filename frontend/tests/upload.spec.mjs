import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const imagePath = new URL(
  "../public/samples/generated-lakehouse.png",
  import.meta.url,
);
const axePath = createRequire(import.meta.url).resolve("axe-core/axe.min.js");
const health = {
  status: "ok",
  image_model: true,
  audio_model: false,
  explainer: false,
  cache: "memory",
  max_upload_bytes: 26214400,
  video_fps: 1,
  max_video_frames: 32,
};
function verdict(score = 0.8) {
  return {
    verdict: score >= 0.5 ? "fake" : "real",
    confidence: Math.max(score, 1 - score),
    fake_probability: score,
    modality: "image",
    model: "browser-test.onnx",
    model_info: {
      id: "Illustrative browser test model",
      source_type: "custom",
      artifact_sha256: "a".repeat(64),
    },
    decision_threshold: 0.5,
    latency_ms: 12,
    cached: false,
    content_sha256: "b".repeat(64),
  };
}
async function respond(route, body, status = 200) {
  await route.fulfill({
    status,
    json: body,
    headers: { "access-control-allow-origin": "*" },
  });
}
async function start(page, healthOverrides = {}) {
  await page.route("**/health", (route) =>
    respond(route, { ...health, ...healthOverrides }),
  );
  await page.goto("/");
  await expect(page.getByText("Detector ready", { exact: true })).toBeVisible();
}
async function choose(page, name = "sample.png") {
  await page.locator('input[type="file"]').setInputFiles({
    name,
    mimeType: "image/png",
    buffer: await readFile(imagePath),
  });
}

test("keyboard selection previews before explicit analysis and exposes the real response fields", async ({
  page,
}) => {
  let requests = 0;
  await page.route("**/verify*", async (route) => {
    requests++;
    await respond(route, verdict());
  });
  await start(page);
  const picker = page.waitForEvent("filechooser");
  await page
    .getByRole("button", { name: "Choose a file", exact: true })
    .press("Enter");
  await (await picker).setFiles(fileURLToPath(imagePath));
  await expect(
    page.getByText("generated-lakehouse.png", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Analyze file", exact: true }),
  ).toBeEnabled();
  expect(requests).toBe(0);
  await page.getByRole("button", { name: "Analyze file", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Likely AI-generated" }),
  ).toBeVisible();
  await expect(page.getByRole("meter")).toHaveAttribute("aria-valuenow", "80");
  await expect(
    page.getByRole("heading", { name: "Likely AI-generated" }),
  ).toBeFocused();
  await page
    .locator("summary")
    .filter({ hasText: "Model & analysis details" })
    .click();
  await expect(
    page.getByText("Illustrative browser test model", { exact: true }),
  ).toBeVisible();
  expect(requests).toBe(1);
});

test("a file dropped on the upload surface reaches the same preview flow", async ({
  page,
}) => {
  await start(page);
  const payload = Array.from(
    Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aP1sAAAAASUVORK5CYII=",
      "base64",
    ),
  );
  const transfer = await page.evaluateHandle((bytes) => {
    const transfer = new DataTransfer();
    transfer.items.add(
      new File([new Uint8Array(bytes)], "dropped.png", { type: "image/png" }),
    );
    return transfer;
  }, payload);
  await page
    .locator(".drop-area")
    .dispatchEvent("drop", { dataTransfer: transfer });
  await expect(page.getByText("dropped.png", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Analyze file", exact: true }),
  ).toBeEnabled();
  await transfer.dispose();
});

test("oversized files are rejected before inference while a valid file can proceed", async ({
  page,
}) => {
  let requests = 0;
  await page.route("**/verify*", (route) => {
    requests++;
    return respond(route, verdict());
  });
  await start(page, { max_upload_bytes: 1024 });
  await choose(page);
  await expect(
    page.getByRole("region", { name: "Media analysis" }).getByRole("alert"),
  ).toContainText("too large");
  expect(requests).toBe(0);
  await page.route("**/health", (route) => respond(route, health));
  await page.reload();
  await expect(page.getByText("Detector ready", { exact: true })).toBeVisible();
  await choose(page);
  await page.getByRole("button", { name: "Analyze file", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Analyze again" }),
  ).toBeVisible();
  expect(requests).toBe(1);
});

test("cancelled work cannot replace a newer file or its result", async ({
  page,
}) => {
  let release;
  const blocked = new Promise((resolve) => {
    release = resolve;
  });
  let calls = 0;
  await page.route("**/verify*", async (route) => {
    calls++;
    if (calls === 1) {
      await blocked;
      await respond(route, verdict(0.95)).catch(() => {});
    } else await respond(route, verdict(0.15));
  });
  try {
    await start(page);
    await choose(page, "first.png");
    await page
      .getByRole("button", { name: "Analyze file", exact: true })
      .click();
    await expect.poll(() => calls).toBe(1);
    await expect(
      page.getByRole("button", { name: "Cancel analysis" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Cancel analysis" }).click();
    await expect(
      page.getByRole("region", { name: "Media analysis" }).getByRole("alert"),
    ).toContainText("cancelled");
    await choose(page, "replacement.png");
    await page
      .getByRole("button", { name: "Analyze file", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Likely camera-made" }),
    ).toBeVisible();
    release();
    await expect(page.getByRole("meter")).toHaveAttribute(
      "aria-valuenow",
      "15",
    );
    await expect(
      page.getByText("replacement.png", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("region", { name: "Media analysis" }).getByRole("alert"),
    ).toHaveCount(0);
  } finally {
    release();
  }
});

test("a decode error stays an error and retry can produce a result", async ({
  page,
}) => {
  let attempts = 0;
  await page.route("**/verify*", (route) =>
    ++attempts === 1
      ? respond(
          route,
          {
            detail:
              "Could not decode this image. Choose a readable image file.",
          },
          422,
        )
      : respond(route, verdict()),
  );
  await start(page);
  await choose(page);
  await page.getByRole("button", { name: "Analyze file", exact: true }).click();
  await expect(
    page.getByRole("region", { name: "Media analysis" }).getByRole("alert"),
  ).toContainText("Could not decode");
  await expect(page.getByRole("meter")).toHaveCount(0);
  await page.getByRole("button", { name: "Analyze file", exact: true }).click();
  await expect(page.getByRole("meter")).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Media analysis" }).getByRole("alert"),
  ).toHaveCount(0);
});

test("unavailable audio capability disables only audio analysis", async ({
  page,
}) => {
  await start(page);
  await page.locator('input[type="file"]').setInputFiles({
    name: "voice.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("test audio"),
  });
  await expect(page.getByRole("status")).toContainText(
    "audio detector is not installed",
  );
  await expect(
    page.getByRole("button", { name: "Analyze file", exact: true }),
  ).toBeDisabled();
  await choose(page);
  await expect(
    page.getByRole("button", { name: "Analyze file", exact: true }),
  ).toBeEnabled();
});

test("an unreachable API can be retried after it recovers", async ({
  page,
}) => {
  let ready = false;
  await page.route("**/health", (route) =>
    ready ? respond(route, health) : route.abort("connectionrefused"),
  );
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText("unavailable");
  ready = true;
  await page.getByRole("button", { name: "Retry connection" }).click();
  await expect(page.getByText("Detector ready", { exact: true })).toBeVisible();
  await expect(page.getByRole("status")).toHaveCount(0);
});

test("mobile themes and long filenames stay within the viewport with reduced motion", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await start(page);
  await choose(page, "a-very-long-field-record-name-".repeat(4) + ".png");
  expect(
    await page.evaluate(
      () => matchMedia("(prefers-reduced-motion: reduce)").matches,
    ),
  ).toBe(true);
  for (const theme of ["dark", "light"]) {
    await page.getByRole("button", { name: "Switch color theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await expect(
      page.getByRole("button", { name: "Analyze file", exact: true }),
    ).toBeVisible();
  }
});

test("media controls update the picker and cancel the previous selection", async ({
  page,
}) => {
  await start(page);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "AI image detector",
  );
  await choose(page);
  await page.getByRole("button", { name: "Video", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "AI video detector",
  );
  await expect(
    page.getByRole("button", { name: "Video", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('input[type="file"]')).toHaveAttribute(
    "accept",
    /\.mp4/,
  );
  await expect(
    page.getByRole("button", { name: "Analyze file", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Try a sample video" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Image", exact: true }).press("Enter");
  await expect(page.locator('input[type="file"]')).toHaveAttribute(
    "accept",
    /\.png/,
  );
  await expect(
    page.getByRole("button", { name: "Choose a file", exact: true }),
  ).toBeVisible();
});

test("sample images are previewed without automatically running inference", async ({
  page,
}) => {
  let requests = 0;
  await page.route("**/verify*", (route) => {
    requests++;
    return respond(route, verdict());
  });
  await start(page);
  await page
    .getByRole("button", { name: "AI-generated image", exact: true })
    .click();
  await expect(
    page.getByText("generated-lakehouse.png", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Analyze file", exact: true }),
  ).toBeEnabled();
  expect(requests).toBe(0);
  await expect(page.getByRole("meter")).toHaveCount(0);
});

test("switching media while an example loads ignores its late response", async ({
  page,
}) => {
  let release;
  const blocked = new Promise((resolve) => {
    release = resolve;
  });
  await page.route("**/samples/generated-lakehouse.png", async (route) => {
    await blocked;
    await route.fulfill({
      body: await readFile(imagePath),
      contentType: "image/png",
    });
  });
  try {
    await start(page);
    await page
      .getByRole("button", { name: "AI-generated image", exact: true })
      .click();
    await expect(page.getByText("Loading example…")).toBeVisible();
    await page.getByRole("button", { name: "Video", exact: true }).click();
    release();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(
      "AI video detector",
    );
    await expect(
      page.getByRole("button", { name: "Try a sample video" }),
    ).toBeEnabled();
    await expect(
      page.getByRole("button", { name: "Analyze file", exact: true }),
    ).toHaveCount(0);
  } finally {
    release();
  }
});

test("empty and result screens fit supported breakpoints in both themes", async ({
  page,
}) => {
  await page.route("**/verify*", (route) => respond(route, verdict()));
  await start(page);
  for (const width of [320, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    for (const selected of [false, true]) {
      if (selected) {
        await choose(page, "a-long-filename-".repeat(12) + ".png");
        await page
          .getByRole("button", { name: "Analyze file", exact: true })
          .click();
        await expect(page.getByRole("meter")).toBeVisible();
      }
      for (const theme of ["light", "dark"]) {
        if (
          ((await page.locator("html").getAttribute("data-theme")) ||
            "light") !== theme
        )
          await page
            .getByRole("button", { name: "Switch color theme" })
            .click();
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBe(true);
      }
    }
    await page
      .getByRole("button", { name: "Remove file", exact: true })
      .click();
  }
});

test("a native request timeout shows an error without a success score", async ({
  page,
}) => {
  // Shorten only this isolated browser's native timer; production remains 180 seconds.
  await page.addInitScript(() => {
    const open = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function (...args) {
      open.apply(this, args);
      this.timeout = 200;
    };
  });
  let release;
  const blocked = new Promise((resolve) => {
    release = resolve;
  });
  await page.route("**/verify*", async (route) => {
    await blocked;
    await respond(route, verdict()).catch(() => {});
  });
  try {
    await start(page);
    await choose(page);
    await page
      .getByRole("button", { name: "Analyze file", exact: true })
      .click();
    await expect(
      page.getByRole("region", { name: "Media analysis" }).getByRole("alert"),
    ).toContainText("timed out");
    await expect(page.getByRole("meter")).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Analyze file", exact: true }),
    ).toBeEnabled();
  } finally {
    release();
  }
});

test("upload and result screens pass automated accessibility checks in both themes", async ({
  page,
}) => {
  await page.route("**/verify*", (route) => respond(route, verdict()));
  await start(page);
  await page.addScriptTag({ path: axePath });
  for (const selected of [false, true]) {
    if (selected) {
      await choose(page);
      await page
        .getByRole("button", { name: "Analyze file", exact: true })
        .click();
      await expect(page.getByRole("meter")).toBeVisible();
    }
    for (const theme of ["light", "dark"]) {
      if (
        ((await page.locator("html").getAttribute("data-theme")) || "light") !==
        theme
      )
        await page.getByRole("button", { name: "Switch color theme" }).click();
      const violations = await page.evaluate(async () => {
        const result = await window.axe.run(document, {
          runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] },
        });
        return result.violations.map(({ id, impact, nodes }) => ({
          id,
          impact,
          targets: nodes.map((node) => node.target),
        }));
      });
      expect(violations).toEqual([]);
    }
  }
});
