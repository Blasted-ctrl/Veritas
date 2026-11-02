import Image from "next/image";
import { IconArrowUpRight, IconPlayerPlay } from "@tabler/icons-react";

type Props = {
  mode: "image" | "video";
  busy: boolean;
  onSelect: (name: string, label: string) => void;
};

export function UploadExamples({ mode, busy, onSelect }: Props) {
  return (
    <aside className="examples" aria-label="Try an example">
      <div className="examples-heading">
        <span>No file handy?</span>
        <span>
          Try an example <IconArrowUpRight size={15} />
        </span>
      </div>
      <div
        className={"example-grid " + (mode === "video" ? "video-example" : "")}
      >
        <button
          className="example-button"
          aria-label={
            mode === "video" ? "Try a sample video" : "AI-generated image"
          }
          disabled={busy}
          onClick={() =>
            onSelect(
              mode === "video"
                ? "generated-demo.mp4"
                : "generated-lakehouse.png",
              mode === "video"
                ? "Video fixture made from an AI-generated still. This tests frame analysis, not AI motion detection."
                : "AI-generated example. Made with OpenAI image generation.",
            )
          }
        >
          <Image
            src="/samples/generated-lakehouse.png"
            alt="Lake house surrounded by mountains"
            fill
            sizes="(max-width: 768px) 45vw, 240px"
          />
          {mode === "video" && (
            <span className="example-play" aria-hidden="true">
              <IconPlayerPlay size={25} />
            </span>
          )}
          <span className="example-label">
            {mode === "video" ? "Try a sample video" : "AI-generated image"}
            <IconArrowUpRight size={16} />
          </span>
        </button>
        {mode === "image" && (
          <button
            className="example-button"
            aria-label="NASA photograph"
            disabled={busy}
            onClick={() =>
              onSelect(
                "earthrise.jpg",
                "NASA photograph. Earthrise, Apollo 8, 1968. Credit: NASA.",
              )
            }
          >
            <Image
              src="/samples/earthrise.jpg"
              alt="Earth rising above the Moon"
              fill
              sizes="(max-width: 768px) 45vw, 240px"
            />
            <span className="example-label">
              NASA photograph
              <IconArrowUpRight size={16} />
            </span>
          </button>
        )}
      </div>
      <p className="examples-note">
        {busy
          ? "Loading example…"
          : "Choose an example, then run your own analysis."}
      </p>
    </aside>
  );
}
