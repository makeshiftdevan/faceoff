/* FaceOff — client-side face blurring. No servers, no uploads, no tracking. */

// Everything is vendored locally (see vendor/NOTICE.txt) — the app makes zero
// third-party requests. Face detection is YuNet (opencv_zoo, MIT) running on
// ONNX Runtime Web's WASM engine.
import * as ort from "./vendor/ort/ort.wasm.bundle.min.mjs";

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

// Detection: frames are downscaled so their long side is at most this before
// running YuNet (dims padded up to multiples of 32, the model's max stride).
const DETECT_MAX_SIDE = 960;
const DETECT_SCORE = 0.6;   // minimum detection confidence
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

let yunet = null; // ORT InferenceSession
let detectorReady = false;
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

async function loadDetector() {
  try {
    yunet = await ort.InferenceSession.create("./assets/face_detection_yunet_2026may.onnx", {
      executionProviders: ["wasm"],
    });
    // Warm up so the first real frame doesn't pay the compile cost.
    const warm = document.createElement("canvas");
    warm.width = 96;
    warm.height = 96;
    warm.getContext("2d").fillRect(0, 0, 96, 96);
    await detectYuNet(warm, {});
    detectorReady = true;
    modelStatus.textContent = "Face detector ready.";
  } catch (err) {
    console.error("Detector load failed:", err);
    modelStatus.textContent =
      "Could not load the face detector. Refresh the page to retry.";
  }
}

// Run YuNet on a canvas. Returns boxes in the source canvas's coordinates:
// { originX, originY, width, height, score }.
//
// Pre/post-processing follows OpenCV's FaceDetectorYN
// (modules/objdetect/src/face_detect.cpp): BGR float input at native scale,
// per-stride grids where score = sqrt(cls * obj), box center = (cell +
// offset) * stride, size = exp(regression) * stride, then greedy IoU NMS.
async function detectYuNet(source, scratch) {
  const sw = source.width;
  const sh = source.height;
  const scale = Math.min(1, DETECT_MAX_SIDE / Math.max(sw, sh));
  const dw0 = Math.max(1, Math.round(sw * scale));
  const dh0 = Math.max(1, Math.round(sh * scale));
  const dw = Math.ceil(dw0 / 32) * 32;
  const dh = Math.ceil(dh0 / 32) * 32;

  if (!scratch.detectCanvas) scratch.detectCanvas = document.createElement("canvas");
  const dc = scratch.detectCanvas;
  if (dc.width !== dw || dc.height !== dh) {
    dc.width = dw;
    dc.height = dh;
  }
  const dctx = dc.getContext("2d", { willReadFrequently: true });
  dctx.fillStyle = "#000";
  dctx.fillRect(0, 0, dw, dh);
  dctx.drawImage(source, 0, 0, sw, sh, 0, 0, dw0, dh0);

  const rgba = dctx.getImageData(0, 0, dw, dh).data;
  const n = dw * dh;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i] = rgba[i * 4 + 2];         // B
    input[n + i] = rgba[i * 4 + 1];     // G
    input[2 * n + i] = rgba[i * 4];     // R
  }
  const feeds = {};
  feeds[yunet.inputNames[0]] = new ort.Tensor("float32", input, [1, 3, dh, dw]);
  const out = await yunet.run(feeds);

  const candidates = [];
  for (const stride of [8, 16, 32]) {
    const cls = out["cls_" + stride].data;
    const obj = out["obj_" + stride].data;
    const bbox = out["bbox_" + stride].data;
    const rows = dh / stride;
    const cols = dw / stride;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const clsScore = Math.min(1, Math.max(0, cls[idx]));
        const objScore = Math.min(1, Math.max(0, obj[idx]));
        const score = Math.sqrt(clsScore * objScore);
        if (score < DETECT_SCORE) continue;
        const cx = (c + bbox[idx * 4]) * stride;
        const cy = (r + bbox[idx * 4 + 1]) * stride;
        const w = Math.exp(bbox[idx * 4 + 2]) * stride;
        const h = Math.exp(bbox[idx * 4 + 3]) * stride;
        candidates.push({
          score,
          originX: cx - w / 2,
          originY: cy - h / 2,
          width: w,
          height: h,
        });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of candidates) {
    if (kept.length >= 100) break;
    if (kept.every((k) => iou(k, box) <= NMS_IOU)) kept.push(box);
  }

  const back = 1 / (dw0 / sw); // undo the downscale
  return kept.map((b) => ({
    score: b.score,
    originX: b.originX * back,
    originY: b.originY * back,
    width: b.width * back,
    height: b.height * back,
  }));
}

async function detectFaces(s) {
  const found = [];
  try {
    const dets = await detectYuNet(s.frameCanvas, s);
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
    ctx.filter = `blur(${radius}px)`;
    ctx.drawImage(s.frameCanvas, 0, 0);
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

  const mimeType = pickMimeType();
  if (!mimeType) {
    showUploadError(
      "This browser can't record video (MediaRecorder missing). Use a recent Chrome, Edge, Firefox, or Safari."
    );
    return;
  }

  clearUploadError();

  const coverage = $("coverage-select").value;
  boxPadding = COVERAGE_PADDING[coverage] ?? COVERAGE_PADDING.normal;

  // Fresh hidden <video> per run (a MediaElementSource can only ever be
  // attached to an element once).
  const video = document.createElement("video");
  video.playsInline = true;
  video.preload = "auto";
  video.crossOrigin = "anonymous";
  video.src = currentObjectUrl;

  session = {
    video,
    active: true,
    faceFrames: 0,
    detectMs: 0,
    tracks: [],
    chunks: [],
    recorder: null,
    audioCtx: null,
    mimeType,
    frameCanvas: document.createElement("canvas"),
    tinyCanvas: document.createElement("canvas"),
  };
  const s = session;

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
    s.frameCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    mergeTracks(s, await detectFaces(s), performance.now());
    renderFrame(s);

    // --- audio: route the element's sound into the recording (not speakers)
    const stream = canvas.captureStream(30);
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
    updateProgress(0, s.duration);

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
    finishProcessing(blob, s);
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
  updateProgress(s.video.currentTime, s.duration, s.faceFrames);
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
    mergeTracks(s, found, when);
    s.faceFrames += found.length;
    // Let render callbacks breathe even when inference is very fast.
    await new Promise((r) => setTimeout(r, 15));
  }
}

function updateProgress(current, duration, faceFrames = 0) {
  const pct = duration > 0 && isFinite(duration)
    ? Math.min(100, Math.round((current / duration) * 100))
    : 0;
  $("progress-fill").style.width = pct + "%";
  $("progress-label").textContent = pct + "%";
  $("progress-bar").setAttribute("aria-valuenow", String(pct));
  $("stat-faces").textContent = String(faceFrames);
  $("stat-time").textContent = `${formatTime(current)} / ${formatTime(duration)}`;
}

function finishProcessing(blob, s) {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(blob);

  const isMp4 = s.mimeType.startsWith("video/mp4");
  const ext = isMp4 ? "mp4" : "webm";
  const baseName = (currentFile?.name || "video").replace(/\.[^.]+$/, "");

  $("result-video").src = resultUrl;
  $("final-faces").textContent = String(s.faceFrames);
  $("final-size").textContent = formatBytes(blob.size);
  $("no-faces-note").hidden = s.faceFrames > 0;
  $("webm-note").hidden = isMp4;

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
