import type { Health, Verdict } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/** Upload a file to /verify with progress, returning the verdict. */
export function verify(
  file: File,
  opts: {
    explain?: boolean;
    onProgress?: (pct: number) => void;
    signal?: AbortSignal;
  } = {},
) {
  return new Promise<Verdict>((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const explain = opts.explain ? "?explain=true" : "";

    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    const cleanup = () => opts.signal?.removeEventListener("abort", abort);
    if (opts.signal?.aborted) {
      reject(new DOMException("Cancelled", "AbortError"));
      return;
    }
    opts.signal?.addEventListener("abort", abort, { once: true });
    xhr.timeout = 180_000;
    xhr.onabort = () => {
      cleanup();
      reject(new DOMException("Cancelled", "AbortError"));
    };
    xhr.ontimeout = () => {
      cleanup();
      reject(new ApiError(0, "Analysis timed out. Try a smaller file."));
    };
    xhr.open("POST", `${API_URL}/verify${explain}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && opts.onProgress)
        opts.onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      cleanup();
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as Verdict);
        else
          reject(
            new ApiError(
              xhr.status,
              body?.detail ?? `request failed (${xhr.status})`,
            ),
          );
      } catch {
        reject(new ApiError(xhr.status, `unexpected response (${xhr.status})`));
      }
    };
    xhr.onerror = () => {
      cleanup();
      reject(
        new ApiError(
          0,
          "Could not reach the analysis service. Check your connection and try again.",
        ),
      );
    };
    xhr.send(form);
  });
}

export async function getHealth(): Promise<Health | null> {
  try {
    const r = await fetch(`${API_URL}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(15000),
    });
    if (!r.ok) return null;
    return (await r.json()) as Health;
  } catch {
    return null;
  }
}
