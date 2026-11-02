import { IconAlertTriangle, IconCircleCheck } from "@tabler/icons-react";
import { useEffect, useRef } from "react";
import type { Verdict } from "@/lib/types";

export function VerdictCard({ result }: { result: Verdict }) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    heading.current?.focus();
  }, []);
  const fake = result.verdict === "fake";
  const score = result.fake_probability * 100;
  return (
    <div className="assessment-result">
      <div className={"prediction " + (fake ? "generated" : "natural")}>
        {fake ? <IconAlertTriangle size={25} /> : <IconCircleCheck size={25} />}
        <h3 ref={heading} tabIndex={-1}>
          {fake
            ? "Likely AI-generated"
            : result.modality === "audio"
              ? "Likely human-made"
              : "Likely camera-made"}
        </h3>
      </div>
      <p className="result-context">
        {result.modality === "video"
          ? "Based on sampled image frames."
          : "The model’s prediction for this file."}{" "}
        This does not establish authenticity.
      </p>
      <div className="score-heading">
        <span>AI-generated score</span>
        <strong>
          {score.toFixed(1)}
          <small>%</small>
        </strong>
      </div>
      <div
        className="score-track"
        role="meter"
        aria-label="AI-generated model score"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number(score.toFixed(1))}
      >
        <span style={{ width: score + "%" }} />
        <i style={{ left: result.decision_threshold * 100 + "%" }} />
      </div>
      <div className="score-scale">
        <span>
          {result.modality === "audio"
            ? "Human-made patterns"
            : "Camera-made patterns"}
        </span>
        <span>AI-made patterns</span>
      </div>
      <p className="score-note">
        Model score, not a measured chance that this file is fake.
      </p>
      {result.frames && (
        <div className="frames">
          <div className="frames-heading">
            <strong>{result.frames_analyzed} frames analyzed</strong>
            <span>Average score above</span>
          </div>
          <div className="frame-bars" aria-hidden="true">
            {result.frames.map((frame) => (
              <span
                key={frame.index}
                style={{
                  height: Math.max(3, frame.fake_probability * 100) + "%",
                }}
              />
            ))}
          </div>
          <details>
            <summary>View individual frame scores</summary>
            <table>
              <caption>Frames in sampling order</caption>
              <thead>
                <tr>
                  <th>Sample</th>
                  <th>AI-generated score</th>
                </tr>
              </thead>
              <tbody>
                {result.frames.map((frame) => (
                  <tr key={frame.index}>
                    <td>{frame.index + 1}</td>
                    <td>{(frame.fake_probability * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
          <p className="score-note">
            Sampled frames only. Audio and motion are not analyzed.
          </p>
        </div>
      )}
      <details className="result-metadata">
        <summary>Model &amp; analysis details</summary>
        <dl>
          <dt>Model</dt>
          <dd>{result.model_info.id}</dd>
          <dt>Source</dt>
          <dd>
            {result.model_info.source_type}
            {result.model_info.source_url && (
              <>
                {" "}
                ·{" "}
                <a
                  href={result.model_info.source_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Model card
                </a>
              </>
            )}
          </dd>
          <dt>License</dt>
          <dd>{result.model_info.license ?? "Not specified"}</dd>
          <dt>Decision threshold</dt>
          <dd>{(result.decision_threshold * 100).toFixed(1)}%</dd>
          <dt>Processing</dt>
          <dd>
            {(result.latency_ms / 1000).toFixed(2)} seconds
            {result.cached ? " · cached result" : ""}
          </dd>
          <dt>Model SHA-256</dt>
          <dd>
            <code>{result.model_info.artifact_sha256}</code>
          </dd>
          <dt>File SHA-256</dt>
          <dd>
            <code>{result.content_sha256}</code>
          </dd>
        </dl>
      </details>
    </div>
  );
}
