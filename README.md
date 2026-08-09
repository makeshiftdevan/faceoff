# FaceOff — Face Blur

A single-page web app that blurs faces in videos **entirely in your browser**.
No servers, no uploads, no cookies, no tracking — your video never leaves your
device. Styled as a Windows 95 program: one wizard window, three steps, no
clutter.

## How it works

1. **Choose a video** — drag & drop or browse (MP4, MOV, WebM, etc.). Files
   over 500 MB get a non-blocking "this might be slow" warning (sterner over
   1 GB).
2. **Blurring** — the video plays into a hidden `<video>` element while
   [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
   (a WIDER-Face-trained detector that handles faces from roughly 10px up)
   runs locally on [ONNX Runtime Web](https://onnxruntime.ai/)'s WASM engine.
   Each frame is downscaled to at most 960px on its long side, run through the
   model (pre/post-processing ported from OpenCV's `FaceDetectorYN`: per-stride
   grids, score = √(cls·obj), IoU NMS), and detections below 0.6 confidence or
   implausibly huge at modest confidence are discarded. A small IoU tracker
   carries each face across brief detection dropouts. Faces get an elliptical
   blur sized by the "Blur size" setting (pixelation fallback on browsers
   without canvas filters), and the blurred canvas + original audio are
   re-encoded live via `MediaRecorder`, with a live preview and progress bar.
3. **Done** — download an MP4 (Chrome, Edge, Safari) or WebM (Firefox, which
   can't write MP4 — the app says so), named `<original>-blurred.mp4`.

Everything is static — no build step, no dependencies to install. The
detector, its WASM runtime, the model, and the UI stylesheet are all
**vendored into this repo** (`vendor/`, `assets/` — see `vendor/NOTICE.txt`),
so the site makes zero requests to third parties. The only optional exception
is a feedback submission, which goes out via Formspree.

Honest limitation: tiny or heavily-shadowed background faces can still be
missed, and a face entering the frame may take a beat to be picked up. Review
the output before sharing anything sensitive.

## Running it

Serve the folder over HTTP (ES modules don't run from `file://`):

```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

Or just enable **GitHub Pages** on this repo — it's a static site.

## Setup you'll want to do

- **Feedback form**: create a free form at [formspree.io](https://formspree.io),
  then replace `YOUR_FORM_ID` at the top of `app.js` with your form id. Until
  then, submissions fall back to opening the visitor's email app (`mailto:`).
- **Donation link**: already wired to <https://paypal.me/dskapetis>.

## Browser support

| Feature | Chrome / Edge | Safari | Firefox |
|---|---|---|---|
| Face detection & blurring | ✅ | ✅ | ✅ |
| MP4 output | ✅ (126+) | ✅ | ❌ → WebM fallback |
| Audio preserved | ✅ | ✅ | ✅ |

Works on desktop and mobile. Processing runs at roughly the playback speed of
the video, on the user's own hardware; slower devices produce a choppier (but
never unblurred) result.

## Privacy

- No backend of any kind — the site is static files, and even the
  face-detection engine is served from the site itself (no CDNs).
- Video data is processed with the Canvas/WebAudio/MediaRecorder APIs and
  never transmitted anywhere.
- No cookies, no storage, no analytics, no ad networks.
- Feedback goes straight through Formspree to email; nothing is stored by
  this site.
