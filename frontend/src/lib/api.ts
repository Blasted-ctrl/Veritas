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
export function verify(file: File, opts: { explain?: boolean; onProgress?: (pct: number) => void } = {}) {
  return new Promise<Verdict>((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const explain = opts.explain ? "?explain=true" : "";

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/verify${explain}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && opts.onProgress) opts.onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as Verdict);
        else reject(new ApiError(xhr.status, body?.detail ?? `request failed (${xhr.status})`));
      } catch {
        reject(new ApiError(xhr.status, `unexpected response (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "network error — is the API running?"));
    xhr.send(form);
  });
}

export async function getHealth(): Promise<Health | null> {
  try {
    const r = await fetch(`${API_URL}/health`, { cache: "no-store" });
    if (!r.ok) return null;
    return (await r.json()) as Health;
  } catch {
    return null;
  }
}
