"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  IconArrowUpRight,
  IconArrowRight,
  IconUpload,
  IconPhoto,
  IconVideo,
  IconX,
  IconSunMoon,
  IconCircleCheck,
  IconAlertTriangle,
  IconPlayerPause,
  IconRefresh,
} from "@tabler/icons-react";
import { getHealth, verify } from "@/lib/api";
import type { Health, Verdict } from "@/lib/types";
import { VerdictCard } from "@/components/VerdictCard";

const ACCEPT =
  ".png,.jpg,.jpeg,.webp,.bmp,.mp4,.mov,.webm,.avi,.mkv,.gif,.wav,.flac,.mp3,.ogg,.m4a";
const VIDEO = /\.(mp4|mov|webm|avi|mkv|gif)$/i;
const AUDIO = /\.(wav|flac|mp3|ogg|m4a)$/i;
const size = (bytes: number) => (bytes / 1024 / 1024).toFixed(1) + " MB";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>();
  const [previewFailed, setPreviewFailed] = useState(false);
  const [result, setResult] = useState<Verdict | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [checking, setChecking] = useState(true);
  const [busy, setBusy] = useState(false);
  const [sampleBusy, setSampleBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);
  const [sample, setSample] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const request = useRef<AbortController | null>(null);
  const serial = useRef(0);
  const objectUrl = useRef<string | undefined>(undefined);

  async function checkHealth() {
    setChecking(true);
    setHealth(await getHealth());
    setChecking(false);
  }
  useEffect(() => {
    let active = true;
    getHealth().then((value) => {
      if (active) {
        setHealth(value);
        setChecking(false);
      }
    });
    return () => {
      active = false;
      request.current?.abort();
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  function selectFile(next: File | null, source = "") {
    serial.current += 1;
    if (input.current) input.current.value = "";
    request.current?.abort();
    setBusy(false);
    setError("");
    setResult(null);
    setProgress(0);
    setSample(source);
    setPreviewFailed(false);
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = undefined;
    if (next && next.size > (health?.max_upload_bytes ?? 26_214_400)) {
      setFile(null);
      setPreview(undefined);
      setError(
        "This file is too large. Choose one under " +
          size(health?.max_upload_bytes ?? 26_214_400) +
          ".",
      );
      return;
    }
    setFile(next);
    if (next) objectUrl.current = URL.createObjectURL(next);
    setPreview(objectUrl.current);
    if (input.current) input.current.value = "";
  }
  async function loadSample(name: string, label: string) {
    const current = ++serial.current;
    setSampleBusy(true);
    setError("");
    try {
      const response = await fetch("/samples/" + name);
      if (!response.ok)
        throw new Error(
          "The sample could not be loaded. Choose a file to continue.",
        );
      const blob = await response.blob();
      if (current === serial.current)
        selectFile(new File([blob], name, { type: blob.type }), label);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load sample.");
    } finally {
      setSampleBusy(false);
    }
  }
  async function analyze() {
    if (!file || busy) return;
    const current = ++serial.current;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    setResult(null);
    setProgress(0);
    try {
      const value = await verify(file, {
        signal: controller.signal,
        onProgress: setProgress,
      });
      if (current === serial.current) setResult(value);
    } catch (e) {
      if (
        current === serial.current &&
        !(e instanceof DOMException && e.name === "AbortError")
      )
        setError(
          e instanceof Error ? e.message : "Analysis failed. Please try again.",
        );
    } finally {
      if (current === serial.current) setBusy(false);
    }
  }
  function cancel() {
    serial.current += 1;
    request.current?.abort();
    setBusy(false);
    setError("Analysis cancelled. Your file is ready to try again.");
  }
  function toggleTheme() {
    const dark = theme
      ? theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    setTheme(next);
  }
  const modality =
    file && AUDIO.test(file.name)
      ? "audio"
      : file && VIDEO.test(file.name)
        ? "video"
        : "image";
  const unavailable =
    !!health &&
    !(modality === "audio" ? health.audio_model : health.image_model);
  const ready = health?.image_model;

  return (
    <div className="site-shell">
      <a className="skip-link" href="#analyze">
        Skip to analysis
      </a>
      <header className="site-header">
        <Link className="wordmark" href="/">
          <span className="brand-mark" aria-hidden="true">v</span>veritas
          <span className="brand-period" aria-hidden="true">.</span>
        </Link>
        <nav aria-label="Main navigation">
          <a href="#how-it-works">How it works</a>
          <a href="#about-model">About the model</a>
          <a
            href="https://github.com/Blasted-ctrl/Veritas"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
            <IconArrowUpRight size={16} />
          </a>
        </nav>
        <button
          className="icon-button theme-button"
          aria-label="Switch color theme"
          onClick={toggleTheme}
        >
          <IconSunMoon size={21} />
        </button>
      </header>
      <main>
        <section
          className={"hero " + (file ? "has-file" : "")}
          aria-labelledby="main-title"
        >
          <div className="introduction">
            <h1 id="main-title">
              Check what
              <br className="title-break" /> you’re seeing.
            </h1>
            <p className="intro-copy">
              Upload an image or video. Take a closer look with an open source
              AI detector.
            </p>
            {!file && (
              <p className="intro-note">
                <IconAlertTriangle size={19} />
                <span>
                  A prediction, not proof.
                  <br />
                  Always check the source, too.
                </span>
              </p>
            )}
          </div>
          <section
            id="analyze"
            className={"analysis-panel " + (file ? "selected" : "")}
            aria-label="Media analysis"
          >
            <div className="panel-heading">
              <h2>{file ? "Your media" : "Analyze your media"}</h2>
              <span className="service-state">
                <span className={"status-dot " + (ready ? "online" : "")} />
                {checking
                  ? "Connecting"
                  : ready
                    ? "Detector ready"
                    : "Detector offline"}
              </span>
            </div>
            <input
              ref={input}
              id="media-file"
              tabIndex={-1}
              className="sr-only"
              type="file"
              accept={ACCEPT}
              aria-label="Choose an image or video"
              onChange={(e) => {
                if (e.target.files?.[0]) selectFile(e.target.files[0]);
              }}
            />
            {!file ? (
              <>
                <div
                  className={"drop-area " + (dragging ? "dragging" : "")}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragging(false);
                    if (e.dataTransfer.files[0])
                      selectFile(e.dataTransfer.files[0]);
                  }}
                >
                  <span className="upload-symbol">
                    <IconUpload size={30} stroke={1.5} />
                  </span>
                  <h3>A little context starts here.</h3>
                  <p>Drop your file here, or choose one below.</p>
                  <button
                    className="button primary"
                    onClick={() => input.current?.click()}
                  >
                    Choose a file
                    <IconArrowRight size={18} />
                  </button>
                  <span className="file-hint">
                    JPG, PNG, WebP, MP4, MOV, WebM + more
                    <br />
                    Up to {size(health?.max_upload_bytes ?? 26_214_400)} per
                    file
                  </span>
                </div>
                <div className="sample-list">
                  <p>Or take a look at an example</p>
                  <div>
                    <button
                      className="sample-button"
                      disabled={sampleBusy}
                      onClick={() =>
                        loadSample(
                          "generated-lakehouse.png",
                          "AI-generated example. Made with OpenAI image generation.",
                        )
                      }
                    >
                      <Image
                        src="/samples/generated-lakehouse.png"
                        alt=""
                        width={48}
                        height={40}
                      />
                      <span>
                        Generated image<small>AI-made landscape</small>
                      </span>
                      <IconArrowUpRight size={17} />
                    </button>
                    <button
                      className="sample-button"
                      disabled={sampleBusy}
                      onClick={() =>
                        loadSample(
                          "earthrise.jpg",
                          "NASA photograph. Earthrise, Apollo 8, 1968. Credit: NASA.",
                        )
                      }
                    >
                      <Image
                        src="/samples/earthrise.jpg"
                        alt=""
                        width={48}
                        height={40}
                      />
                      <span>
                        NASA photograph<small>Earthrise, Apollo 8</small>
                      </span>
                      <IconArrowUpRight size={17} />
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="workbench">
                <div className="source-pane">
                  <div className="media-preview">
                    {previewFailed ? (
                      <p className="preview-unavailable">
                        Preview unavailable. You can try analyzing the file, or
                        choose another one.
                      </p>
                    ) : (
                      preview &&
                      (modality === "video" && !/\.gif$/i.test(file.name) ? (
                        <video
                          src={preview}
                          controls
                          preload="metadata"
                          onError={() => setPreviewFailed(true)}
                        />
                      ) : modality === "audio" ? (
                        <audio
                          src={preview}
                          controls
                          onError={() => setPreviewFailed(true)}
                        />
                      ) : (
                        <Image
                          src={preview}
                          alt={"Preview of " + file.name}
                          fill
                          unoptimized
                          onError={() => setPreviewFailed(true)}
                          sizes="(max-width: 720px) 100vw, 50vw"
                          style={{ objectFit: "contain" }}
                        />
                      ))
                    )}
                  </div>
                  <div className="file-details">
                    <span className="file-icon">
                      {modality === "video" ? <IconVideo /> : <IconPhoto />}
                    </span>
                    <div>
                      <strong title={file.name}>{file.name}</strong>
                      <span>
                        {size(file.size)} ·{" "}
                        {modality === "video"
                          ? "Video frames"
                          : modality === "audio"
                            ? "Audio"
                            : "Image"}
                      </span>
                    </div>
                    <button
                      className="icon-button"
                      aria-label="Remove file"
                      onClick={() => selectFile(null)}
                    >
                      <IconX size={19} />
                    </button>
                  </div>
                  {sample && (
                    <p className="sample-caption">
                      {sample}{" "}
                      <a href="/samples/credits.json">
                        Source details
                        <IconArrowUpRight size={12} />
                      </a>
                    </p>
                  )}
                  <button
                    className="text-button"
                    onClick={() => input.current?.click()}
                  >
                    Choose a different file
                  </button>
                </div>
                <div
                  className="assessment-pane"
                  aria-live="polite"
                  aria-atomic="false"
                >
                  {result ? (
                    <VerdictCard result={result} />
                  ) : (
                    <div className="assessment-empty">
                      <span className="assessment-icon">
                        {busy ? (
                          <IconRefresh size={28} />
                        ) : (
                          <IconPhoto size={28} />
                        )}
                      </span>
                      <h3>
                        {busy
                          ? progress < 100
                            ? "Uploading your file"
                            : "Taking a closer look"
                          : "Ready for a closer look?"}
                      </h3>
                      <p>
                        {busy
                          ? "The detector is checking visual patterns. Larger videos take longer."
                          : modality === "video"
                            ? "We’ll sample frames and show each image score alongside the average."
                            : "Run the detector to see its prediction, score, and model details."}
                      </p>
                      {busy && (
                        <>
                          <progress
                            aria-label="Upload progress"
                            max={100}
                            value={progress}
                          />
                          <span className="file-hint">
                            {progress < 100
                              ? progress + "% uploaded"
                              : "Upload complete · analyzing"}
                          </span>
                        </>
                      )}
                    </div>
                  )}
                  <div className="analysis-actions">
                    {busy ? (
                      <button className="button secondary" onClick={cancel}>
                        <IconPlayerPause size={17} />
                        Cancel analysis
                      </button>
                    ) : (
                      <button
                        className="button primary"
                        onClick={analyze}
                        disabled={checking || !health || unavailable}
                      >
                        {result ? "Analyze again" : "Analyze file"}
                        <IconArrowRight size={18} />
                      </button>
                    )}
                  </div>
                  {!busy && !result && (
                    <p className="analysis-disclaimer">
                      Scores can be wrong, even when they are high.
                    </p>
                  )}
                </div>
              </div>
            )}
            {error && (
              <div className="notice error" role="alert">
                <IconAlertTriangle size={19} />
                <p>{error}</p>
              </div>
            )}
            {!checking && (!health || unavailable) && (
              <div className="notice" role="status">
                <IconAlertTriangle size={19} />
                <p>
                  {!health
                    ? "The analysis service is unavailable. Please try again shortly."
                    : "The " +
                      modality +
                      " detector is not installed on this server."}
                </p>
                <button className="text-button" onClick={checkHealth}>
                  Retry connection
                </button>
              </div>
            )}
            <div className="privacy-note">
              <IconCircleCheck size={16} />
              <span>
                Files are processed by this server. Uploads are not saved;
                scores may be cached.
              </span>
            </div>
          </section>
        </section>
        <section id="how-it-works" className="explainer-section">
          <h2>
            A second look,
            <br />
            with the context intact.
          </h2>
          <div className="explanation">
            <p>
              Veritas was built to make AI media detection easier to try and
              understand. You bring the file. The detector looks for patterns
              learned from real and generated images.
            </p>
            <p>
              For video, it checks a sample of frames and averages their scores.
              It does not verify audio, edits, or the story a video tells.
            </p>
            <a
              className="inline-link"
              href="https://github.com/Blasted-ctrl/Veritas#run-it-locally"
            >
              Run it on your own machine
              <IconArrowUpRight size={17} />
            </a>
          </div>
        </section>
        <section id="about-model" className="model-section">
          <div>
            <h2>
              Know the model.
              <br />
              Know its limits.
            </h2>
            <a
              className="inline-link"
              href="https://huggingface.co/onnx-community/ai-image-detection-ONNX"
              target="_blank"
              rel="noreferrer"
            >
              Read the model card
              <IconArrowUpRight size={17} />
            </a>
          </div>
          <div className="model-details">
            <details open>
              <summary>What powers the default detector?</summary>
              <p>
                A pretrained Vision Transformer trained on CIFAKE images, served
                locally with ONNX Runtime. Each result identifies the actual
                model used. Operators can also bring their own trained model.
              </p>
            </details>
            <details>
              <summary>Can it tell me whether a file is authentic?</summary>
              <p>
                No detector can establish authenticity on its own. This model
                learned from a limited set of older generated images. New
                generators, compression, screenshots, and unfamiliar subjects
                can confuse it. A high score is not proof.
              </p>
            </details>
            <details>
              <summary>What happens to my upload?</summary>
              <p>
                Image bytes are processed in memory. Videos use a temporary file
                that is deleted after analysis. The service caches scores by
                file hash, and may cache an optional explanation if requested. A
                hosted operator controls its own infrastructure.
              </p>
            </details>
          </div>
        </section>
      </main>
      <footer className="site-footer">
        <Link className="wordmark small" href="/">
          veritas.
        </Link>
        <span>Stay curious. Check the source.</span>
        <a href="https://github.com/Blasted-ctrl/Veritas">
          Open source on GitHub
          <IconArrowUpRight size={15} />
        </a>
      </footer>
    </div>
  );
}
