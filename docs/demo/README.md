# Veritas demo

[Download the captioned MP4](https://github.com/Blasted-ctrl/Veritas/raw/refs/heads/main/docs/demo/veritas-demo.mp4) · [Contact sheet](contact-sheet.jpg)

GitHub shows the looping GIF in the main README. Download the MP4 to play
the complete demo; GitHub's file viewer does not play this video inline.

44.5 seconds · 1920 × 1080 · 24 fps · Silent, with instructions on screen.

The demo records the working website at its actual speed. It shows NASA's
Earthrise photograph, a labeled generated landscape, and a four-second MP4
made from that generated still image. The clip demonstrates frame sampling;
it is not independently generated moving footage.

The displayed scores are real API responses from the artwork EfficientNet-B0
CNN, using the conservative decision cutoff **0.8770313858985902**. Some
responses come from the application's normal result cache, which the UI
identifies. No requests were intercepted and no verdicts were fabricated.
[recording.json](recording.json) preserves those responses and the graph checksum.

The examples demonstrate the product flow. They do not measure detector
accuracy. At this cutoff the CNN misses many AI images. Scores are class
scores, not probabilities of authenticity. The Docker quickstart downloads
a separate ViT model. [Model evaluation](../evaluation/artwork-v3/README.md).

## Reproduce the recording

Start the website and API using [the development guide](../development.md).
Use your chosen model and cutoff; your scores may differ. Then run:

```sh
cd frontend
npm ci
npx playwright install chromium
node scripts/record-demo.mjs --url http://127.0.0.1:3000 --output ../.turbo/demo-recording
```

The script records real uploads in a fresh browser context for each sample,
checks the displayed scores against the actual API responses, and saves
WebM clips plus a response manifest. It uses no API mocks.

## Edit or render the film

[source/](source/) includes the HyperFrames composition, original recordings,
timing, local fonts, and rendering dependencies. Node 22+, FFmpeg, and an
installed Chromium browser are required. The project pins HyperFrames 0.9.4.

```sh
cd docs/demo/source
npm run dev
```

To render:

```sh
npm run check
npm run render -- --quality looks --fps 24 --output ../veritas-demo.mp4
```

If the CLI cannot find Chromium, set `HYPERFRAMES_BROWSER_PATH` to its
executable. Set `HYPERFRAMES_CACHE_DIR` and `--frames-cache-dir` to a drive
with enough free space if needed. The source recordings are not stretched
or sped up; the title and closing cards are added around them. The film is
silent, with instructions on screen. The automated checker reports no
lint, runtime, layout, or contrast errors; it warns about the deliberate
silent soundtrack and the composition's flat section structure.

## Credits

- Earthrise: NASA / William Anders, Apollo 8, 1968. [NASA source](https://science.nasa.gov/resource/image-earthrise/).
- Generated landscape: OpenAI image generation, created for the Veritas demo.
- Sample MP4: the generated still encoded with FFmpeg. [Sample provenance](../../frontend/public/samples/credits.json).
- Hanken fonts: [SIL Open Font License](source/assets/OFL.txt).
- GSAP 3.15.0: copyright 2026 GreenSock, used under its [standard license](https://gsap.com/standard-license). The original notice is retained in the bundled script.
