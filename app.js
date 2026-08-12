/* FaceOff — client-side face blurring. No servers, no uploads, no tracking. */

// Everything is vendored locally (see vendor/NOTICE.txt) — the app makes zero
// third-party requests. Face detection is YuNet (opencv_zoo, MIT) running on
// ONNX Runtime Web's WASM engine, entirely inside a worker (detect-worker.js)
// so the main thread — which draws and records every frame — never blocks.
//
// Two processing modes:
//  - Precise (default, needs WebCodecs): every source frame is decoded,
//    scanned, blurred, and re-encoded with its exact timestamp via
//    Mediabunny. Slower than the video, but frame-perfect: full source
//    frame rate, and the blur is measured on the very frame it covers.
//  - Realtime (fallback): the video plays into a canvas that is recorded
//    with MediaRecorder while detection tracks faces in parallel.
import {
  Input,
  BlobSource,
  ALL_FORMATS,
  Output,
  BufferTarget,
  Mp4OutputFormat,
  VideoSampleSink,
  CanvasSource,
  AudioSampleSink,
  AudioSampleSource,
  EncodedPacketSink,
  EncodedAudioPacketSource,
  getFirstEncodableVideoCodec,
  getFirstEncodableAudioCodec,
  QUALITY_HIGH,
} from "./vendor/mediabunny/mediabunny.min.mjs";

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------

// Replace with your real Formspree form id (e.g. "mzbqwxyz") to activate
// the feedback form. Until then, submissions fall back to a mailto: link.
const FORMSPREE_ID = "YOUR_FORM_ID";
const FALLBACK_EMAIL = "devanskapetis@gmail.com";

const WARN_BYTES = 500 * 1024 * 1024;      // gentle warning above 500 MB
const BIG_WARN_BYTES = 1024 * 1024 * 1024; // stronger wording above 1 GB
const MAX_DIMENSION = 1920;                // cap output resolution for sanity

// Detection: frames are downscaled so their long side matches the current
// rung of this ladder before running YuNet (dims padded up to multiples of
// 32, the model's max stride). The loop steps down when inference is too
// slow for the device and back up when there's headroom — higher resolution
// means smaller faces get caught.
const DETECT_SIDES = [1280, 960, 768, 640];
const DETECT_SLOW_MS = 400; // step resolution down above this per-pass time
const DETECT_FAST_MS = 150; // step back up below this
const DETECT_SCORE = 0.5;   // minimum detection confidence (recall-first)
const NMS_IOU = 0.3;        // overlap threshold for non-max suppression
const MATCH_IOU = 0.25;     // overlap needed to treat a detection as the same face

// Rendering runs at full video frame rate; detection runs in parallel at
// whatever pace the device manages. Between detections, tracked boxes are
// advanced along their measured velocity and slightly inflated with age so
// the blur stays on a moving face.
const TRACK_KEEP_MS = 900;      // drop a track not re-detected for this long
const MAX_EXTRAPOLATE_S = 0.4;  // cap on motion extrapolation
const STALE_GROW = 0.5;         // extra box growth per second of staleness

// A "face" covering a huge chunk of the frame at modest confidence is a
// hallucination (a real close-up face scores high, so those still pass).
const HUGE_BOX_AREA = 0.35;
const HUGE_BOX_MIN_SCORE = 0.8;

// Precise mode: with detection running on every single frame, a lost face
// only needs a short linger (in frames) to bridge detector blinks.
const OFFLINE_LINGER_FRAMES = 3;
// A face can move far between frames, dropping box overlap to zero, so
// detections are matched against each track's velocity-predicted position
// and accepted when the centre lands within this multiple of the box size.
// Too small and fast motion spawns duplicate tracks that linger as ghost
// blurs trailing the real face.
const MATCH_CENTRE_DIST = 2.0;

// How much to expand the detected face box before blurring.
const COVERAGE_PADDING = { snug: 0.0, normal: 0.1, extra: 0.3 };
let boxPadding = COVERAGE_PADDING.normal;

const MIME_CANDIDATES = [
  'video/mp4;codecs="avc1.42E01E,mp4a.40.2"',
  "video/mp4",
  'video/webm;codecs="vp9,opus"',
  'video/webm;codecs="vp8,opus"',
  "video/webm",
];

// ------------------------------------------------------------------
// DOM
// ------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

const screens = {
  upload: $("screen-upload"),
  processing: $("screen-processing"),
  download: $("screen-download"),
};

const dropZone = $("drop-zone");
const fileInput = $("file-input");
const browseBtn = $("browse-btn");
const sizeWarning = $("size-warning");
const sizeWarningText = $("size-warning-text");
const uploadError = $("upload-error");
const fileReady = $("file-ready");
const startBtn = $("start-btn");
const cancelBtn = $("cancel-btn");
const againBtn = $("again-btn");
const modelStatus = $("model-status");

const canvas = $("work-canvas");
const ctx = canvas.getContext("2d");

// ------------------------------------------------------------------
// State
// ------------------------------------------------------------------

let detectWorker = null;
let detectorReady = false;
let detectReqId = 0;
const detectPending = new Map();
let currentFile = null;
let currentObjectUrl = null;
let resultUrl = null;

let session = null; // active processing session

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function showScreen(name) {
  for (const [key, el] of Object.entries(screens)) el.hidden = key !== name;
  window.scrollTo({ top: 0 });
}

function formatBytes(bytes) {
  if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  return Math.max(1, Math.round(bytes / 1024)) + " KB";
}

// Some files (notably WebMs written by screen recorders) report Infinity as
// their duration until you seek to the end once. Standard workaround.
function resolveDuration(video) {
  if (isFinite(video.duration)) return Promise.resolve(video.duration);
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      video.currentTime = 0;
      resolve(video.duration);
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = Number.MAX_SAFE_INTEGER;
    setTimeout(() => {
      video.removeEventListener("seeked", onSeeked);
      resolve(video.duration);
    }, 3000);
  });
}

function formatTime(seconds) {
  if (!isFinite(seconds)) return "?:??";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function showUploadError(message) {
  uploadError.textContent = "❌ " + message;
  uploadError.hidden = false;
}

function clearUploadError() {
  uploadError.hidden = true;
  uploadError.textContent = "";
}

function pickMimeType() {
  if (typeof MediaRecorder === "undefined") return null;
  for (const mime of MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported(mime)) return mime;
  }
  return null;
}

function canvasFilterSupported() {
  const testCtx = document.createElement("canvas").getContext("2d");
  testCtx.filter = "blur(2px)";
  return testCtx.filter === "blur(2px)";
}
const useCanvasBlur = canvasFilterSupported();

// ------------------------------------------------------------------
// Face detector: YuNet on ONNX Runtime Web (WASM, runs locally)
// ------------------------------------------------------------------

function loadDetector() {
  try {
    detectWorker = new Worker(new URL("detect-worker.js", document.baseURI), { type: "module" });
    detectWorker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "ready") {
        detectorReady = true;
        modelStatus.textContent = "Face detector ready.";
        return;
      }
      const req = detectPending.get(msg.id);
      if (!req) return;
      detectPending.delete(msg.id);
      if (msg.type === "result") req.resolve(msg.boxes);
      else req.reject(new Error(msg.message || "Detection failed."));
    };
    detectWorker.onerror = (e) => {
      console.error("Detector worker failed:", e.message || e);
      if (!detectorReady) {
        modelStatus.textContent = "Could not load the face detector. Refresh the page to retry.";
      }
    };
    detectWorker.postMessage({
      type: "init",
      wasmPaths: new URL("vendor/ort/", document.baseURI).href,
      modelUrl: new URL("assets/face_detection_yunet_2026may.onnx", document.baseURI).href,
    });
  } catch (err) {
    console.error("Detector load failed:", err);
    modelStatus.textContent = "Could not load the face detector. Refresh the page to retry.";
  }
}

// Hand the current frame to the worker (zero-copy bitmap transfer) and get
// back boxes in canvas coordinates.
async function detectFaces(s) {
  const found = [];
  try {
    const bitmap = await createImageBitmap(s.frameCanvas);
    const dets = await new Promise((resolve, reject) => {
      const id = ++detectReqId;
      detectPending.set(id, { resolve, reject });
      detectWorker.postMessage(
        {
          type: "detect",
          id,
          bitmap,
          maxSide: DETECT_SIDES[s.detectSideIdx],
          score: DETECT_SCORE,
          nmsIou: NMS_IOU,
        },
        [bitmap]
      );
    });
    for (const box of dets) {
      const areaFrac = (box.width * box.height) / (canvas.width * canvas.height);
      if (areaFrac > HUGE_BOX_AREA && box.score < HUGE_BOX_MIN_SCORE) continue;
      found.push(box);
    }
  } catch (err) {
    // A single bad frame shouldn't kill the run.
    console.warn("Detection hiccup:", err);
  }
  return found;
}

function iou(a, b) {
  const x1 = Math.max(a.originX, b.originX);
  const y1 = Math.max(a.originY, b.originY);
  const x2 = Math.min(a.originX + a.width, b.originX + b.width);
  const y2 = Math.min(a.originY + a.height, b.originY + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  return inter / (a.width * a.height + b.width * b.height - inter);
}

// Merge fresh detections into tracked faces, measuring per-track velocity
// so the render loop can extrapolate positions between detections.
function mergeTracks(s, boxes, when) {
  for (const box of boxes) {
    let best = null;
    let bestIou = MATCH_IOU;
    for (const t of s.tracks) {
      const i = iou(t.box, box);
      if (i > bestIou) {
        best = t;
        bestIou = i;
      }
    }
    if (best) {
      const dt = (when - best.lastSeen) / 1000;
      if (dt > 0.005) {
        const vx = (box.originX - best.box.originX) / dt;
        const vy = (box.originY - best.box.originY) / dt;
        best.vx = 0.5 * best.vx + 0.5 * vx;
        best.vy = 0.5 * best.vy + 0.5 * vy;
      }
      best.box = box;
      best.lastSeen = when;
    } else {
      s.tracks.push({ box, vx: 0, vy: 0, lastSeen: when });
    }
  }
}

// Precise-mode tracker: detection runs on every frame, so tracking is just
// "blur what was found, and let a lost face linger a few frames, slightly
// grown, to bridge detector blinks".
function offlineMerge(s, boxes) {
  for (const t of s.tracks) {
    t.missed = (t.missed ?? 0) + 1;
    // Where this face should be now, given how it was moving.
    t.predicted = shiftBox(t.box, t.vx * t.missed, t.vy * t.missed);
  }
  for (const box of boxes) {
    let best = null;
    let bestScore = 0;
    for (const t of s.tracks) {
      if (t.claimed) continue;
      const score = matchScore(t.predicted, box);
      if (score > bestScore) {
        best = t;
        bestScore = score;
      }
    }
    if (best) {
      const frames = best.missed;
      best.vx = 0.6 * best.vx + 0.4 * (box.originX - best.box.originX) / frames;
      best.vy = 0.6 * best.vy + 0.4 * (box.originY - best.box.originY) / frames;
      best.claimed = true;
      best.box = box;
      best.missed = 0;
    } else {
      // claimed so a second detection in this same frame can't match it,
      // and predicted so it is a complete track if that comparison happens
      s.tracks.push({ box, predicted: box, vx: 0, vy: 0, missed: 0, claimed: true });
    }
  }
  for (const t of s.tracks) t.claimed = false;
  s.tracks = s.tracks.filter((t) => t.missed <= OFFLINE_LINGER_FRAMES);
}

function shiftBox(b, dx, dy) {
  return { originX: b.originX + dx, originY: b.originY + dy, width: b.width, height: b.height };
}

// How strongly two boxes look like the same face: overlap, or failing that,
// proximity of their centres relative to their size (which survives the
// large jumps a fast-moving face makes between frames).
function matchScore(a, b) {
  const overlap = iou(a, b);
  if (overlap > MATCH_IOU) return 1 + overlap;
  const ax = a.originX + a.width / 2;
  const ay = a.originY + a.height / 2;
  const bx = b.originX + b.width / 2;
  const by = b.originY + b.height / 2;
  const size = (a.width + a.height + b.width + b.height) / 4;
  if (size <= 0) return 0;
  const dist = Math.hypot(ax - bx, ay - by) / size;
  // Similar sizes too — a distant face shouldn't absorb a near one.
  const ratio = Math.min(a.width / b.width, b.width / a.width);
  if (dist > MATCH_CENTRE_DIST || ratio < 0.5) return 0;
  return 1 - dist / MATCH_CENTRE_DIST;
}

// While a face is briefly un-detected, keep its blur moving along the path
// it was travelling and grow it a little to stay covered.
function offlineBox(t) {
  if (!t.missed) return t.box;
  const moved = shiftBox(t.box, t.vx * t.missed, t.vy * t.missed);
  const grow = 1 + 0.15 * t.missed;
  const w = moved.width * grow;
  const h = moved.height * grow;
  return {
    originX: moved.originX + moved.width / 2 - w / 2,
    originY: moved.originY + moved.height / 2 - h / 2,
    width: w,
    height: h,
  };
}

// Where a track's blur should be drawn right now: last detected box, moved
// along its velocity and grown a little the longer it hasn't been confirmed.
function predictBox(t, now) {
  const age = Math.min((now - t.lastSeen) / 1000, MAX_EXTRAPOLATE_S);
  const grow = 1 + STALE_GROW * age;
  const w = t.box.width * grow;
  const h = t.box.height * grow;
  const cx = t.box.originX + t.box.width / 2 + t.vx * age;
  const cy = t.box.originY + t.box.height / 2 + t.vy * age;
  return { originX: cx - w / 2, originY: cy - h / 2, width: w, height: h };
}

// ------------------------------------------------------------------
// File selection
// ------------------------------------------------------------------

function handleFile(file) {
  clearUploadError();
  sizeWarning.hidden = true;
  fileReady.hidden = true;

  if (!file) return;

  const looksLikeVideo =
    (file.type && file.type.startsWith("video/")) ||
    /\.(mp4|mov|avi|mkv|webm|m4v|3gp|mpg|mpeg|ogv)$/i.test(file.name);
  if (!looksLikeVideo) {
    showUploadError(`"${file.name}" is not a video file. Use MP4, MOV, or WebM.`);
    return;
  }

  if (file.size > WARN_BYTES) {
    sizeWarningText.textContent =
      file.size > BIG_WARN_BYTES
        ? `This file is ${formatBytes(file.size)} (over 1 GB). Processing happens on this device and may be slow or fail, especially on phones. You can still continue.`
        : `This file is ${formatBytes(file.size)}. Processing happens on this device and may be slow. You can still continue.`;
    sizeWarning.hidden = false;
  }

  currentFile = file;
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = URL.createObjectURL(file);

  // Probe the file with a throwaway <video> to catch unsupported codecs early
  // and to read the duration for display.
  const probe = document.createElement("video");
  probe.preload = "metadata";
  probe.muted = true;
  probe.src = currentObjectUrl;

  probe.onloadedmetadata = async () => {
    const duration = await resolveDuration(probe);
    $("file-name").textContent = file.name;
    $("file-size").textContent = formatBytes(file.size);
    $("file-duration").textContent = formatTime(duration);
    fileReady.hidden = false;
    probe.removeAttribute("src");
    probe.load();
  };
  probe.onerror = () => {
    currentFile = null;
    showUploadError(
      `This browser can't play "${file.name}". Convert it to MP4 (H.264) and try again.`
    );
  };
}

browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (e) => {
  if (e.target !== browseBtn) fileInput.click();
});
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  handleFile(file);
});

// ------------------------------------------------------------------
// Processing pipeline
//   hidden <video> -> face detection -> blurred canvas ->
//   canvas.captureStream + WebAudio audio -> MediaRecorder
// ------------------------------------------------------------------

function blurRegion(s, box) {
  // box is in canvas coordinates. Expand by the chosen coverage margin.
  const padW = box.width * boxPadding;
  const padH = box.height * boxPadding;
  const x = box.originX - padW / 2;
  const y = box.originY - padH / 2;
  const w = box.width + padW;
  const h = box.height + padH;

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
  ctx.clip();

  if (useCanvasBlur) {
    const radius = Math.max(10, Math.round(w / 8));
    // Draw only the patch around the face (padded by 2x the blur radius so
    // the kernel's edge falloff never reaches the visible ellipse) instead
    // of pushing the whole frame through the filter for every face.
    const m = radius * 2;
    const px = Math.max(0, x - m);
    const py = Math.max(0, y - m);
    const pw = Math.min(canvas.width, x + w + m) - px;
    const ph = Math.min(canvas.height, y + h + m) - py;
    ctx.filter = `blur(${radius}px)`;
    ctx.drawImage(s.frameCanvas, px, py, pw, ph, px, py, pw, ph);
    ctx.filter = "none";
  } else {
    // Fallback for browsers without canvas filters: chunky pixelation.
    // Sample the face region from the clean frame, shrink it way down,
    // then stretch it back up with smoothing off.
    const block = Math.max(8, Math.round(w / 10));
    const tinyW = Math.max(2, Math.round(w / block));
    const tinyH = Math.max(2, Math.round(h / block));
    const tiny = s.tinyCanvas;
    tiny.width = tinyW;
    tiny.height = tinyH;
    const tctx = tiny.getContext("2d");
    tctx.drawImage(s.frameCanvas, x, y, w, h, 0, 0, tinyW, tinyH);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tiny, 0, 0, tinyW, tinyH, x, y, w, h);
    ctx.imageSmoothingEnabled = true;
  }
  ctx.restore();
}

async function startProcessing() {
  if (!currentFile || !currentObjectUrl) return;
  if (!detectorReady) {
    showUploadError("The face detector is still loading. Try again in a moment.");
    return;
  }

  clearUploadError();

  const coverage = $("coverage-select").value;
  boxPadding = COVERAGE_PADDING[coverage] ?? COVERAGE_PADDING.normal;

  session = {
    video: null,
    active: true,
    faceFrames: 0,
    detectMs: 0,
    detectSideIdx: 0,
    tracks: [],
    chunks: [],
    recorder: null,
    audioCtx: null,
    mimeType: null,
    startedAt: performance.now(),
    frameCanvas: document.createElement("canvas"),
    tinyCanvas: document.createElement("canvas"),
  };
  const s = session;

  // Precise mode: frame-exact processing via WebCodecs. Falls back to the
  // realtime pipeline when the browser or the file can't do it.
  if (typeof VideoDecoder !== "undefined" && typeof VideoEncoder !== "undefined") {
    try {
      document.body.dataset.mode = "precise";
      await processPrecise(s);
      return;
    } catch (err) {
      if (!s.active) return; // cancelled mid-run
      console.warn("Precise mode unavailable, using realtime fallback:", err);
      document.body.dataset.mode = "realtime";
      s.tracks = [];
      s.faceFrames = 0;
      s.detectSideIdx = 0;
      s.startedAt = performance.now();
    }
  }
  await processRealtime(s);
}

// ------------------------------------------------------------------
// Precise mode: decode every source frame, scan it, blur it, re-encode it
// with its original timestamp, and copy the audio track untouched.
// ------------------------------------------------------------------

async function processPrecise(s) {
  const input = new Input({ source: new BlobSource(currentFile), formats: ALL_FORMATS });
  try {
    const vTrack = await input.getPrimaryVideoTrack();
    if (!vTrack) throw new Error("no video track");
    if (!(await vTrack.canDecode())) throw new Error("codec not decodable via WebCodecs");
    const duration = await input.computeDuration();
    s.duration = duration;

    // Size the canvas (cap the longest side to keep memory + encoder happy;
    // even dimensions keep H.264 encoders happy).
    let w = vTrack.displayWidth;
    let h = vTrack.displayHeight;
    const longest = Math.max(w, h);
    if (longest > MAX_DIMENSION) {
      const scale = MAX_DIMENSION / longest;
      w = Math.round(w * scale);
      h = Math.round(h * scale);
    }
    canvas.width = w - (w % 2);
    canvas.height = h - (h % 2);
    s.frameCanvas.width = canvas.width;
    s.frameCanvas.height = canvas.height;
    s.frameCtx = s.frameCanvas.getContext("2d");

    const outputFormat = new Mp4OutputFormat({ fastStart: "in-memory" });
    const videoCodec = await getFirstEncodableVideoCodec(
      ["avc", "hevc", "vp9", "av1"].filter((c) => outputFormat.getSupportedCodecs().includes(c)),
      { width: canvas.width, height: canvas.height }
    );
    if (!videoCodec) throw new Error("no encodable MP4 video codec");

    const output = new Output({ format: outputFormat, target: new BufferTarget() });
    const videoSource = new CanvasSource(canvas, { codec: videoCodec, quality: QUALITY_HIGH });
    output.addVideoTrack(videoSource);

    // Audio: copy the original packets untouched when MP4 can hold them;
    // otherwise re-encode; otherwise ship without audio and say so.
    let audio = null;
    s.audioDropped = false;
    const aTrack = await input.getPrimaryAudioTrack();
    if (aTrack) {
      if (aTrack.codec && outputFormat.getSupportedCodecs().includes(aTrack.codec)) {
        const src = new EncodedAudioPacketSource(aTrack.codec);
        output.addAudioTrack(src);
        audio = {
          kind: "copy",
          src,
          iter: new EncodedPacketSink(aTrack).packets(),
          meta: { decoderConfig: await aTrack.getDecoderConfig() },
          pending: null,
          done: false,
        };
      } else if (await aTrack.canDecode()) {
        const aCodec = await getFirstEncodableAudioCodec(
          ["aac", "opus"].filter((c) => outputFormat.getSupportedCodecs().includes(c)),
          { numberOfChannels: aTrack.numberOfChannels, sampleRate: aTrack.sampleRate }
        );
        if (aCodec) {
          const src = new AudioSampleSource({ codec: aCodec, quality: QUALITY_HIGH });
          output.addAudioTrack(src);
          audio = {
            kind: "samples",
            src,
            iter: new AudioSampleSink(aTrack).samples(),
            pending: null,
            done: false,
          };
        } else {
          s.audioDropped = true;
        }
      } else {
        s.audioDropped = true;
      }
    }

    await output.start();
    showScreen("processing");
    updateProgress(0, duration, 0, s);

    const vSink = new VideoSampleSink(vTrack);
    for await (const sample of vSink.samples()) {
      if (!s.active) {
        sample.close();
        await output.cancel();
        return;
      }
      const ts = sample.timestamp;
      const dur = sample.duration;
      sample.draw(s.frameCtx, 0, 0, canvas.width, canvas.height);
      sample.close();

      // Detect on this exact frame — no prediction, no staleness.
      const found = await detectFaces(s);
      offlineMerge(s, found);
      s.faceFrames += found.length;

      ctx.drawImage(s.frameCanvas, 0, 0);
      for (const t of s.tracks) blurRegion(s, offlineBox(t));

      await videoSource.add(ts, dur); // built-in encoder backpressure
      await pumpAudio(audio, ts + 1);
      updateProgress(ts, duration, s.faceFrames, s);
    }

    if (!s.active) {
      await output.cancel();
      return;
    }
    await pumpAudio(audio, Infinity);
    await output.finalize();

    const blob = new Blob([output.target.buffer], { type: "video/mp4" });
    finishProcessing(blob, s, { isMp4: true });
  } finally {
    try {
      input.dispose();
    } catch {}
  }
}

// Feed audio into the muxer up to the given media timestamp, keeping the
// file nicely interleaved as video frames are appended.
async function pumpAudio(audio, untilTs) {
  if (!audio || audio.done) return;
  while (true) {
    if (!audio.pending) {
      const r = await audio.iter.next();
      if (r.done) {
        audio.done = true;
        return;
      }
      audio.pending = r.value;
    }
    if (audio.pending.timestamp > untilTs) return;
    const item = audio.pending;
    audio.pending = null;
    if (audio.kind === "copy") {
      await audio.src.add(item, audio.meta);
    } else {
      await audio.src.add(item);
      item.close();
    }
  }
}

// ------------------------------------------------------------------
// Realtime fallback: play the video into the canvas and record it while
// detection tracks faces in parallel.
// ------------------------------------------------------------------

async function processRealtime(s) {
  document.body.dataset.mode = document.body.dataset.mode || "realtime";
  const mimeType = pickMimeType();
  if (!mimeType) {
    failProcessing(
      new Error(
        "This browser can't process video (WebCodecs and MediaRecorder both missing). Use a recent Chrome, Edge, Firefox, or Safari."
      )
    );
    return;
  }
  s.mimeType = mimeType;

  // Fresh hidden <video> per run (a MediaElementSource can only ever be
  // attached to an element once).
  const video = document.createElement("video");
  video.playsInline = true;
  video.preload = "auto";
  video.crossOrigin = "anonymous";
  video.src = currentObjectUrl;
  s.video = video;

  try {
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = () => reject(new Error("This video can't be decoded by your browser."));
    });
    s.duration = await resolveDuration(video);

    // Size the canvas (cap the longest side to keep memory + encoder happy).
    let w = video.videoWidth;
    let h = video.videoHeight;
    const longest = Math.max(w, h);
    if (longest > MAX_DIMENSION) {
      const scale = MAX_DIMENSION / longest;
      w = Math.round(w * scale);
      h = Math.round(h * scale);
    }
    // Even dimensions keep H.264 encoders happy.
    canvas.width = w - (w % 2);
    canvas.height = h - (h % 2);
    s.frameCanvas.width = canvas.width;
    s.frameCanvas.height = canvas.height;
    s.frameCtx = s.frameCanvas.getContext("2d");

    // Detect + render the first frame BEFORE recording starts, so not even
    // one unblurred frame can slip into the output.
    if (video.readyState < 2) {
      await new Promise((resolve) => {
        video.oncanplay = resolve;
        setTimeout(resolve, 3000);
      });
    }
    // Run a few passes on the opening frame: single-pass recall on small
    // faces is imperfect, and this frame doubles as the video's poster.
    s.frameCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    for (let pass = 0; pass < 3; pass++) {
      const before = s.tracks.length;
      mergeTracks(s, await detectFaces(s), performance.now());
      s.faceFrames += Math.max(0, s.tracks.length - before);
      if (pass > 0 && s.tracks.length === before) break; // stable
    }
    for (const t of s.tracks) t.lastSeen = performance.now(); // don't age out during setup
    renderFrame(s);

    // --- audio: route the element's sound into the recording (not speakers)
    // No frame-rate argument: capture every canvas paint, so 60 fps sources
    // record at 60 fps when the device keeps up.
    const stream = canvas.captureStream();
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      s.audioCtx = new AudioCtx();
      await s.audioCtx.resume();
      const source = s.audioCtx.createMediaElementSource(video);
      const dest = s.audioCtx.createMediaStreamDestination();
      source.connect(dest); // deliberately NOT connected to speakers
      dest.stream.getAudioTracks().forEach((t) => stream.addTrack(t));
    } catch (err) {
      console.warn("Audio capture unavailable, recording video only:", err);
    }

    s.recorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: 12_000_000,
    });
    s.recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) s.chunks.push(e.data);
    };
    s.recorder.onerror = (e) => failProcessing(e.error || new Error("Recording failed."));

    const finished = new Promise((resolve) => (s.recorder.onstop = resolve));

    showScreen("processing");
    updateProgress(0, s.duration, 0, s);

    s.recorder.start(1000);
    await video.play();

    // --- render loop: every video frame, at full rate (cheap draws only)
    const onFrame = () => {
      if (!s.active) return;
      renderFrame(s);
      scheduleFrame();
    };
    const scheduleFrame = () => {
      if (!s.active) return;
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
      else requestAnimationFrame(onFrame);
    };
    scheduleFrame();

    // --- detection loop: runs in parallel at its own pace
    detectionLoop(s);

    await new Promise((resolve) => {
      video.onended = resolve;
      video.onerror = () => resolve(); // decode hiccup mid-file: keep what we have
    });
    if (!s.active) return; // cancelled

    s.active = false;
    if (s.recorder.state !== "inactive") s.recorder.stop();
    await finished;

    const blob = new Blob(s.chunks, { type: mimeType.split(";")[0] });
    finishProcessing(blob, s, { isMp4: mimeType.startsWith("video/mp4") });
  } catch (err) {
    failProcessing(err);
  } finally {
    if (s.audioCtx) s.audioCtx.close().catch(() => {});
    video.removeAttribute("src");
    video.load();
  }
}

// Draw the current video frame with blurs at the tracks' predicted
// positions. Runs at full video frame rate; must stay cheap.
function renderFrame(s) {
  const now = performance.now();
  s.frameCtx.drawImage(s.video, 0, 0, canvas.width, canvas.height);
  // On slow devices a single detection pass can take a while; never expire
  // tracks faster than the detector can re-confirm them.
  const keepMs = Math.max(TRACK_KEEP_MS, (s.detectMs || 0) * 3);
  s.tracks = s.tracks.filter((t) => now - t.lastSeen < keepMs);
  ctx.drawImage(s.frameCanvas, 0, 0);
  for (const t of s.tracks) blurRegion(s, predictBox(t, now));
  updateProgress(s.video.currentTime, s.duration, s.faceFrames, s);
}

// Detect continuously while processing is active. Each pass reads whatever
// frame the render loop most recently copied into frameCanvas.
async function detectionLoop(s) {
  while (s.active) {
    const when = performance.now();
    const found = await detectFaces(s);
    if (!s.active) return;
    const dur = performance.now() - when;
    s.detectMs = s.detectMs ? 0.7 * s.detectMs + 0.3 * dur : dur;
    // Trade detection resolution against pace to fit this device.
    if (s.detectMs > DETECT_SLOW_MS && s.detectSideIdx < DETECT_SIDES.length - 1) {
      s.detectSideIdx++;
      s.detectMs = 0; // re-measure at the new size
    } else if (s.detectMs && s.detectMs < DETECT_FAST_MS && s.detectSideIdx > 0) {
      s.detectSideIdx--;
      s.detectMs = 0;
    }
    mergeTracks(s, found, when);
    s.faceFrames += found.length;
    // Let render callbacks breathe even when inference is very fast.
    await new Promise((r) => setTimeout(r, 15));
  }
}

function updateProgress(current, duration, faceFrames = 0, s = null) {
  const pct = duration > 0 && isFinite(duration)
    ? Math.min(100, Math.round((current / duration) * 100))
    : 0;
  $("progress-fill").style.width = pct + "%";
  $("progress-label").textContent = pct + "%";
  $("progress-bar").setAttribute("aria-valuenow", String(pct));
  $("stat-faces").textContent = String(faceFrames);
  let time = `${formatTime(current)} / ${formatTime(duration)}`;
  if (s && s.startedAt && pct >= 3 && pct < 100) {
    const elapsed = (performance.now() - s.startedAt) / 1000;
    time += ` · about ${formatTime((elapsed * (100 - pct)) / pct)} left`;
  }
  $("stat-time").textContent = time;
}

function finishProcessing(blob, s, { isMp4 }) {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(blob);

  const ext = isMp4 ? "mp4" : "webm";
  const baseName = (currentFile?.name || "video").replace(/\.[^.]+$/, "");

  $("result-video").src = resultUrl;
  $("final-faces").textContent = String(s.faceFrames);
  $("final-size").textContent = formatBytes(blob.size);
  $("no-faces-note").hidden = s.faceFrames > 0;
  $("webm-note").hidden = isMp4;
  $("audio-note").hidden = !s.audioDropped;

  const link = $("download-link");
  link.href = resultUrl;
  link.download = `${baseName}-blurred.${ext}`;

  showScreen("download");
  session = null;
}

function failProcessing(err) {
  console.error(err);
  if (session) {
    session.active = false;
    try {
      if (session.recorder && session.recorder.state !== "inactive") session.recorder.stop();
    } catch {}
    try {
      session.video.pause();
    } catch {}
    session = null;
  }
  showScreen("upload");
  showUploadError(
    (err && err.message ? err.message + " " : "Processing failed. ") +
      "Nothing left this device. Try a different or smaller video."
  );
}

function cancelProcessing() {
  if (!session) return;
  const s = session;
  s.active = false;
  try {
    s.video.pause();
  } catch {}
  try {
    if (s.recorder && s.recorder.state !== "inactive") s.recorder.stop();
  } catch {}
  if (s.audioCtx) s.audioCtx.close().catch(() => {});
  session = null;
  showScreen("upload");
}

startBtn.addEventListener("click", startProcessing);
cancelBtn.addEventListener("click", cancelProcessing);
againBtn.addEventListener("click", () => {
  $("result-video").removeAttribute("src");
  $("result-video").load();
  if (resultUrl) {
    URL.revokeObjectURL(resultUrl);
    resultUrl = null;
  }
  fileInput.value = "";
  currentFile = null;
  fileReady.hidden = true;
  sizeWarning.hidden = true;
  clearUploadError();
  showScreen("upload");
});

// ------------------------------------------------------------------
// Feedback form (Formspree — nothing stored on any server of ours)
// ------------------------------------------------------------------

const gbForm = $("guestbook-form");
const gbStatus = $("gb-status");

gbForm.action = `https://formspree.io/f/${FORMSPREE_ID}`;

gbForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = new FormData(gbForm);

  if (FORMSPREE_ID === "YOUR_FORM_ID") {
    // Formspree not configured yet — fall back to the visitor's mail app.
    const subject = encodeURIComponent("FaceOff feedback");
    const body = encodeURIComponent(
      `Name: ${data.get("name") || "(anonymous)"}\nEmail: ${data.get("email") || "(none)"}\n\n${data.get("message")}`
    );
    window.location.href = `mailto:${FALLBACK_EMAIL}?subject=${subject}&body=${body}`;
    gbStatus.textContent = "Opening your email app…";
    return;
  }

  gbStatus.textContent = "Sending…";
  try {
    const res = await fetch(gbForm.action, {
      method: "POST",
      body: data,
      headers: { Accept: "application/json" },
    });
    if (res.ok) {
      gbForm.reset();
      gbStatus.textContent = "Sent. Thank you.";
    } else {
      gbStatus.textContent = "Sending failed. Try again later.";
    }
  } catch {
    gbStatus.textContent = "No connection. Try again later.";
  }
});

// ------------------------------------------------------------------
// Boot
// ------------------------------------------------------------------

loadDetector();
