export type Modality = "image" | "audio" | "video";

export interface FrameVerdict {
  index: number;
  fake_probability: number;
}

export interface Verdict {
  verdict: "real" | "fake";
  confidence: number;
  fake_probability: number;
  modality: Modality;
  model: string;
  latency_ms: number;
  cached: boolean;
  content_sha256: string;
  heatmap?: string | null;
  overlay?: string | null;
  frames_analyzed?: number | null;
  frames?: FrameVerdict[] | null;
}

export interface Health {
  status: string;
  image_model: boolean;
  audio_model: boolean;
  explainer: boolean;
  cache: string;
}
