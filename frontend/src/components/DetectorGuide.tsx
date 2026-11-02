import { IconArrowUpRight } from "@tabler/icons-react";

export function DetectorGuide() {
  return (
    <>
      <section
        id="how-it-works"
        className="how-section"
        aria-labelledby="how-title"
      >
        <div className="section-heading">
          <p className="eyebrow">A few simple steps</p>
          <h2 id="how-title">Check before you share.</h2>
          <p>
            From a saved photo to a video in your feed, start with the file.
          </p>
        </div>
        <ol className="steps">
          <li>
            <span className="step-number">1</span>
            <h3>Upload your file</h3>
            <p>
              Choose an image or video, or drag it straight into the detector.
            </p>
          </li>
          <li>
            <span className="step-number">2</span>
            <h3>Run the analysis</h3>
            <p>
              The detector checks for visual patterns learned from real and
              AI-generated images.
            </p>
          </li>
          <li>
            <span className="step-number">3</span>
            <h3>Read your result</h3>
            <p>
              See the prediction and AI score. Use them alongside the original
              source and context.
            </p>
          </li>
        </ol>
      </section>
      <section
        id="about-model"
        className="faq-section"
        aria-labelledby="faq-title"
      >
        <div>
          <p className="eyebrow">Good to know</p>
          <h2 id="faq-title">Frequently asked questions</h2>
          <p className="faq-intro">
            A clearer picture of what the detector can tell you.
          </p>
        </div>
        <div className="model-details">
          <details>
            <summary>How accurate is AI detection?</summary>
            <p>
              Results can be wrong. Real photographs can be flagged as AI, and
              generated images can go undetected. New generators, compression
              and unfamiliar subjects can affect the result. A high score is not
              proof of authenticity or manipulation.
            </p>
          </details>
          <details>
            <summary>What does the AI score mean?</summary>
            <p>
              It shows how strongly the file matches AI-generated patterns
              learned by the model. It is a model score, not a measured chance
              that the file is fake. Each result includes the model and decision
              threshold in its analysis details.
            </p>
          </details>
          <details>
            <summary>How are videos checked?</summary>
            <p>
              Veritas samples image frames and averages their scores. You can
              inspect each frame score in the result. It does not analyze
              motion, speech, lip sync or the story a video tells.
            </p>
          </details>
          <details>
            <summary>What happens to my upload?</summary>
            <p>
              Images are processed in memory. Videos use a temporary file that
              is deleted after analysis. Scores may be cached by file hash.
              Uploads are not used for training by this service; a hosted
              operator controls its own infrastructure.
            </p>
          </details>
          <details>
            <summary>Which model is used?</summary>
            <p>
              The default detector is a pretrained Vision Transformer trained on
              CIFAKE images, served with ONNX Runtime. A result identifies the
              actual model used, including custom models installed by the
              operator.
            </p>
            <a
              className="inline-link"
              href="https://huggingface.co/onnx-community/ai-image-detection-ONNX"
              target="_blank"
              rel="noreferrer"
            >
              Read the default model card <IconArrowUpRight size={16} />
            </a>
          </details>
        </div>
      </section>
    </>
  );
}
