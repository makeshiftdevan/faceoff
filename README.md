# FaceOff — Face Blur

A single-page web app that blurs faces in videos **entirely in your browser**.
No servers, no uploads, no cookies, no tracking — your video never leaves your
device. Styled as a Windows 95 program: one wizard window, three steps, no
clutter.

## How it works

1. **Choose a video** — drag & drop or browse (MP4, MOV, WebM, etc.). Files
   over 500 MB get a non-blocking "this might be slow" warning (sterner over
   1 GB).
2. **Blurring** — faces are found by
   [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
   (a WIDER-Face-trained detector that handles faces from roughly 10px up)
   running locally on [ONNX Runtime Web](https://onnxruntime.ai/)'s WASM
   engine, inside a Web Worker so the UI thread is never blocked. Detection
   pre/post-processing is ported from OpenCV's `FaceDetectorYN` (per-stride
   grids, score = √(cls·obj), IoU NMS, 0.5 confidence floor, and rejection of
   implausibly huge boxes). Detected faces get an elliptical blur sized by
   the "Blur size" setting, with a pixelation fallback on browsers without
   canvas filters.

   There are two pipelines:

   - **Precise** (default, needs WebCodecs — Chrome, Edge, Safari, recent
     Firefox): every source frame is decoded via
     [Mediabunny](https://mediabunny.dev/), scanned, blurred, and re-encoded
     with its original timestamp into an MP4; audio packets are copied
     across untouched. Because nothing is racing a clock, the output keeps
     the source frame rate exactly and each blur is measured on the very
     frame it covers. Processing takes longer than the video's duration.
   - **Realtime** (fallback): the video plays into a canvas recorded with
     `MediaRecorder` while detection runs in parallel, extrapolating each
     face's position between passes.

   A small tracker matches detections across frames using overlap, or a
   velocity-predicted position when a fast-moving face jumps too far for
   boxes to overlap, and briefly lingers a lost face to bridge detector
   blinks.
3. **Done** — download the result, named `<original>-blurred.mp4`. The
   precise pipeline always writes MP4; the realtime fallback writes WebM on
   browsers that can't encode MP4 (the app says so when that happens).

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
| Precise (frame-exact) pipeline | ✅ | ✅ | ✅ (130+) |
| MP4 output | ✅ | ✅ | ✅ precise / WebM fallback |
| Audio preserved | ✅ | ✅ | ✅ |

Works on desktop and mobile. In precise mode processing takes longer than the
video itself — how much longer depends on the device — but the output always
runs at the source frame rate. On the realtime fallback, a slow device
produces a choppier (but never unblurred) result.

## Privacy

- No backend of any kind — the site is static files, and even the
  face-detection engine is served from the site itself (no CDNs).
- Video data is processed with the Canvas/WebAudio/MediaRecorder APIs and
  never transmitted anywhere.
- No cookies, no storage, no analytics, no ad networks.
- Feedback goes straight through Formspree to email; nothing is stored by
  this site.
