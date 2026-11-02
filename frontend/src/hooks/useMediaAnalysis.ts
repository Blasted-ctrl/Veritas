"use client";

import { useEffect, useRef, useState } from "react";
import { getHealth, verify } from "@/lib/api";
import type { Health, Verdict } from "@/lib/types";

const VIDEO = /\.(mp4|mov|webm|avi|mkv|gif)$/i;
const AUDIO = /\.(wav|flac|mp3|ogg|m4a)$/i;
export const formatSize = (bytes: number) =>
  (bytes / 1024 / 1024).toFixed(1) + " MB";
const size = formatSize;

export function useMediaAnalysis() {
  const [mode, setMode] = useState<"image" | "video">("image");
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
    setSampleBusy(false);
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
    if (next) setMode(VIDEO.test(next.name) ? "video" : "image");
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
      if (current === serial.current)
        setError(e instanceof Error ? e.message : "Could not load sample.");
    } finally {
      if (current === serial.current) setSampleBusy(false);
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

  function changeMode(next: "image" | "video") {
    if (next === mode) return;
    selectFile(null);
    setMode(next);
    setSampleBusy(false);
  }
  return {
    mode,
    changeMode,
    file,
    preview,
    previewFailed,
    setPreviewFailed,
    result,
    health,
    checking,
    checkHealth,
    busy,
    sampleBusy,
    progress,
    error,
    dragging,
    setDragging,
    sample,
    input,
    selectFile,
    loadSample,
    analyze,
    cancel,
    toggleTheme,
    modality,
    unavailable,
    ready,
  };
}
