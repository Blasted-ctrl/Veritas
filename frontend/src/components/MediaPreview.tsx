import Image from "next/image";
import {
  IconArrowUpRight,
  IconPhoto,
  IconVideo,
  IconX,
} from "@tabler/icons-react";
import { formatSize } from "@/hooks/useMediaAnalysis";

type Props = {
  file: File;
  preview?: string;
  failed: boolean;
  onError: () => void;
  modality: string;
  sample: string;
  onRemove: () => void;
  onReplace: () => void;
};

export function MediaPreview({
  file,
  preview,
  failed,
  onError,
  modality,
  sample,
  onRemove,
  onReplace,
}: Props) {
  return (
    <div className="source-pane">
      <div className="media-preview">
        {failed ? (
          <p className="preview-unavailable">
            Preview unavailable. You can try analyzing the file, or choose
            another one.
          </p>
        ) : (
          preview &&
          (modality === "video" && !/\.gif$/i.test(file.name) ? (
            <video
              src={preview}
              controls
              preload="metadata"
              onError={onError}
            />
          ) : modality === "audio" ? (
            <audio src={preview} controls onError={onError} />
          ) : (
            <Image
              src={preview}
              alt={"Preview of " + file.name}
              fill
              unoptimized
              onError={onError}
              sizes="(max-width: 768px) 100vw, 50vw"
            />
          ))
        )}
      </div>
      <div className="file-details">
        <span className="file-icon" aria-hidden="true">
          {modality === "video" ? (
            <IconVideo size={22} />
          ) : (
            <IconPhoto size={22} />
          )}
        </span>
        <div>
          <strong title={file.name}>{file.name}</strong>
          <span>
            {formatSize(file.size)} ·{" "}
            {modality === "video"
              ? "Video"
              : modality === "audio"
                ? "Audio"
                : "Image"}
          </span>
        </div>
        <button
          className="icon-button"
          aria-label="Remove file"
          onClick={onRemove}
        >
          <IconX size={20} />
        </button>
      </div>
      {sample && (
        <p className="sample-caption">
          {sample}{" "}
          <a href="/samples/credits.json">
            Source details <IconArrowUpRight size={12} />
          </a>
        </p>
      )}
      <button className="text-button" onClick={onReplace}>
        Choose a different file
      </button>
    </div>
  );
}
