# FaceOff!! 😎 — The Totally Private Face Blurrinator 3000

A single-page web app that blurs faces in videos **entirely in your browser**.
No servers, no uploads, no cookies, no tracking — your video never leaves your
device. Wrapped in a loving tribute to the 1997 web: tiled teal backgrounds,
blinking "under construction" banners, a hit counter, and Comic Sans as far as
the eye can see.

## How it works

1. **Upload** — drag & drop or pick a video (MP4, MOV, WebM, etc.). Files over
   500 MB get a gentle "this might be slow" warning (over 1 GB, a slightly
   sterner one), but nothing is ever blocked.
2. **Processing** — the video plays into a hidden `<video>` element while
   [MediaPipe Face Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)
   (WASM, running locally) finds faces frame by frame. Because the detector
   internally shrinks its input to a small square, small/distant faces would
   vanish on a full-frame pass — so every frame also gets one rotating square
   tile crop (at one or two zoom levels, depending on resolution), with hits
   mapped back to frame coordinates. A tiny IoU tracker holds each face
   between tile revisits and across brief dropouts. Detections below 0.5
   confidence are ignored, and implausibly huge boxes (over a third of the
   frame at modest confidence) are rejected as hallucinations. Each face gets
   an elliptical blur sized by the "blur coverage" setting (pixelation
   fallback on browsers without canvas filters), and the blurred canvas +
   original audio are re-encoded live via `MediaRecorder`. A chunky retro
   progress bar shows live progress, along with a live preview.
3. **Download** — you get an MP4 (Chrome, Edge, Safari) or WebM (Firefox,
   which can't write MP4 — the app tells you when that happens), named
   `<original>-blurred.mp4`.

Honest limitation: very small background faces (a distant crowd) can still
slip past the detector, and a face entering the frame far from the camera may
take a few tenths of a second to be picked up by the tile rotation. Skim the
output before sharing anything sensitive.

Everything is static — no build step, no dependencies to install. The
MediaPipe library, its WASM runtime, and the face model are all **vendored
into this repo** (`vendor/tasks-vision/`, `assets/`), so the site makes zero
requests to third parties. The only optional exception is a guestbook
submission, which goes out via Formspree.

## Running it

Serve the folder over HTTP (ES modules don't run from `file://`):

```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

Or just enable **GitHub Pages** on this repo — it's a static site.

## Setup you'll want to do

- **Guestbook**: create a free form at [formspree.io](https://formspree.io),
  then replace `YOUR_FORM_ID` at the top of `app.js` with your form id. Until
  then, guestbook submissions fall back to opening the visitor's email app
  (`mailto:`) instead — nothing breaks.
- **Donation button**: already wired to <https://paypal.me/dskapetis>.

## Browser support

| Feature | Chrome / Edge | Safari | Firefox |
|---|---|---|---|
| Face detection & blurring | ✅ | ✅ | ✅ |
| MP4 output | ✅ (126+) | ✅ | ❌ → WebM fallback |
| Audio preserved | ✅ | ✅ | ✅ |

Works on desktop and mobile. Processing runs at roughly the playback speed of
the video, on the user's own hardware.

## Privacy

- No backend of any kind — the site is static files, and even the
  face-detection engine is served from the site itself (no CDNs).
- Video data is processed with the Canvas/WebAudio/MediaRecorder APIs and
  never transmitted anywhere.
- No cookies, no localStorage, no analytics, no ad networks. The "hit counter"
  is decorative and computed from the date.
- Guestbook entries go straight through Formspree to email; nothing is stored
  by this site.
