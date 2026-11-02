"use client";

import {
  IconArrowUpRight,
  IconArrowRight,
  IconUpload,
  IconPhoto,
  IconVideo,
  IconLock,
  IconAlertTriangle,
  IconX,
  IconRefresh,
} from "@tabler/icons-react";
import { useMediaAnalysis, formatSize } from "@/hooks/useMediaAnalysis";
import { VerdictCard } from "@/components/VerdictCard";
import { MediaPreview } from "@/components/MediaPreview";
import { UploadExamples } from "@/components/UploadExamples";
import { DetectorGuide } from "@/components/DetectorGuide";
import { Brand } from "@/components/Brand";

export default function Home() {
  const {
    file,
    mode,
    result,
    health,
    busy,
    checking,
    unavailable,
    progress,
    changeMode,
    ready,
    input,
    selectFile,
    dragging,
    setDragging,
    sampleBusy,
    loadSample,
    preview,
    previewFailed,
    setPreviewFailed,
    modality,
    sample,
    cancel,
    analyze,
    error,
    checkHealth,
  } = useMediaAnalysis();
  const maxSize = formatSize(health?.max_upload_bytes ?? 26_214_400);
  return (
    <div className="site-shell">
      <a className="skip-link" href="#analyze">
        Skip to analysis
      </a>
      <header className="site-header">
        <Brand />
        <nav aria-label="Main navigation">
          <a href="#how-it-works">How it works</a>
          <a href="#about-model">FAQs</a>
        </nav>
      </header>
      <main>
        <section className="hero" aria-labelledby="main-title">
          <div className="introduction">
            <h1 id="main-title">AI {mode} detector</h1>
            <p>
              Was it made with AI? Check{" "}
              {mode === "image" ? "an image" : "a video’s frames"} for signs of
              AI generation.
              <br className="desktop-break" /> Upload your file and see what the
              detector finds.
            </p>
          </div>
          <div className="media-selector" role="group" aria-label="Media type">
            <button
              aria-pressed={mode === "image"}
              onClick={() => changeMode("image")}
            >
              <IconPhoto size={19} />
              Image
            </button>
            <button
              aria-pressed={mode === "video"}
              onClick={() => changeMode("video")}
            >
              <IconVideo size={19} />
              Video
            </button>
          </div>
          <section
            id="analyze"
            className={"analysis-panel " + (file ? "selected" : "")}
            aria-label="Media analysis"
          >
            <div className="panel-heading">
              <h2>
                {result
                  ? "Your analysis"
                  : file
                    ? "Your file is ready"
                    : "Check your " + mode}
              </h2>
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
              accept={
                mode === "image"
                  ? ".png,.jpg,.jpeg,.webp,.bmp"
                  : ".mp4,.mov,.webm,.avi,.mkv,.gif"
              }
              aria-label="Choose an image or video"
              onChange={(event) => {
                if (event.target.files?.[0]) selectFile(event.target.files[0]);
              }}
            />
            {!file ? (
              <div className="upload-layout">
                <div
                  className={"drop-area " + (dragging ? "dragging" : "")}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(event) => {
                    event.preventDefault();
                    setDragging(false);
                    if (event.dataTransfer.files[0])
                      selectFile(event.dataTransfer.files[0]);
                  }}
                >
                  <span className="upload-symbol" aria-hidden="true">
                    <IconUpload size={30} />
                  </span>
                  <h3>Drag &amp; drop your {mode} here</h3>
                  <p>or select a file from your device</p>
                  <button
                    className="button primary"
                    onClick={() => input.current?.click()}
                  >
                    Choose a file
                    <IconArrowRight size={18} />
                  </button>
                  <span className="file-hint">
                    {mode === "image"
                      ? "JPG, PNG, WebP or BMP"
                      : "MP4, MOV, WebM, AVI, MKV or GIF"}{" "}
                    · Up to {maxSize}
                  </span>
                </div>
                <UploadExamples
                  mode={mode}
                  busy={sampleBusy}
                  onSelect={loadSample}
                />
              </div>
            ) : (
              <div className="workbench">
                <MediaPreview
                  file={file}
                  preview={preview}
                  failed={previewFailed}
                  onError={() => setPreviewFailed(true)}
                  modality={modality}
                  sample={sample}
                  onRemove={() => selectFile(null)}
                  onReplace={() => input.current?.click()}
                />
                <div
                  className="assessment-pane"
                  aria-live="polite"
                  aria-atomic="false"
                  aria-busy={busy}
                >
                  {result ? (
                    <VerdictCard result={result} />
                  ) : (
                    <div className="assessment-empty">
                      <span className="assessment-icon" aria-hidden="true">
                        {busy ? (
                          <IconRefresh size={26} />
                        ) : (
                          <IconPhoto size={26} />
                        )}
                      </span>
                      <h3>
                        {busy
                          ? progress < 100
                            ? "Uploading your file…"
                            : "Analyzing your file…"
                          : "Ready to check?"}
                      </h3>
                      <p>
                        {busy
                          ? "Looking for AI-generated patterns. Longer videos can take a little more time."
                          : modality === "video"
                            ? "We’ll check sampled frames and show their average AI score."
                            : "Get a prediction and an AI score for your file."}
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
                        <IconX size={18} />
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
                      A model prediction, not proof of authenticity.
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
              <IconLock size={16} />
              <span>
                Uploads are processed by this server and aren’t saved. Scores
                may be cached.
              </span>
            </div>
          </section>
          <p className="tool-note">
            AI detection can make mistakes. Always check the source and context,
            too.
          </p>
        </section>
        <DetectorGuide />
      </main>
      <footer className="site-footer">
        <Brand small />
        <span>Check the media. Keep the context.</span>
        <a
          href="https://github.com/Blasted-ctrl/Veritas"
          target="_blank"
          rel="noreferrer"
        >
          Open source on GitHub
          <IconArrowUpRight size={15} />
        </a>
      </footer>
    </div>
  );
}
