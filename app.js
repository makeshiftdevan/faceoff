/* FaceOff!! — client-side face blurring. No servers, no uploads, no tracking. */

// MediaPipe is vendored locally (see vendor/tasks-vision/NOTICE.txt), so the
// app makes zero third-party requests — everything is served from this site.
import {
  FaceDetector,
  FilesetResolver,
} from "./vendor/tasks-vision/vision_bundle.mjs";

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------

// Replace with your real Formspree form id (e.g. "mzbqwxyz") to activate
// the guestbook. Until then, submissions fall back to a mailto: link.
const FORMSPREE_ID = "YOUR_FORM_ID";
const FALLBACK_EMAIL = "devanskapetis@gmail.com";

const WARN_BYTES = 500 * 1024 * 1024;      // gentle warning above 500 MB
const BIG_WARN_BYTES = 1024 * 1024 * 1024; // stronger wording above 1 GB
const MAX_DIMENSION = 1920;                // cap output resolution for sanity
const BOX_STICKY_FRAMES = 12;              // keep blurring briefly after a lost detection
const BOX_PADDING = 0.28;                  // expand detected boxes by 28%

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

let detector = null;
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
  uploadError.textContent = "💥 " + message;
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
// Face detector (MediaPipe, runs as WASM in the browser)
// ------------------------------------------------------------------

async function loadDetector() {
  try {
    const vision = await FilesetResolver.forVisionTasks("./vendor/tasks-vision/wasm");
    const options = (delegate) => ({
      baseOptions: {
        modelAssetPath: "./assets/blaze_face_short_range.tflite",
        delegate,
      },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.4,
    });
    try {
      detector = await FaceDetector.createFromOptions(vision, options("GPU"));
    } catch (gpuErr) {
      console.warn("GPU delegate unavailable, falling back to CPU:", gpuErr);
      detector = await FaceDetector.createFromOptions(vision, options("CPU"));
    }
    // Warm-up run so the first real frame doesn't pay the shader/graph
    // compilation cost mid-video.
    try {
      const warm = document.createElement("canvas");
      warm.width = 64;
      warm.height = 64;
      warm.getContext("2d").fillRect(0, 0, 64, 64);
      detector.detectForVideo(warm, performance.now());
    } catch {}
    detectorReady = true;
    modelStatus.textContent = "🤖 Face-finding robot is ready to go!";
  } catch (err) {
    console.error("Detector load failed:", err);
    modelStatus.textContent =
      "😵 Couldn't load the face-detection engine. Try refreshing the page — " +
      "if it keeps happening, your browser may be too old for WebAssembly.";
  }
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
    showUploadError(
      `"${file.name}" doesn't look like a video file. Try an MP4, MOV, WebM, or similar.`
    );
    return;
  }

  if (file.size > WARN_BYTES) {
    sizeWarningText.textContent =
      file.size > BIG_WARN_BYTES
        ? ` This video is ${formatBytes(file.size)} — over 1 GB! Since everything runs on your own device, processing may be quite slow or could even fail, especially on phones.`
        : ` This video is ${formatBytes(file.size)}. Since everything runs on your own device, processing might be slow depending on your hardware.`;
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
      `Your browser can't play "${file.name}" — the format or codec isn't supported here. ` +
        "MP4 (H.264), MOV, and WebM almost always work; AVI often doesn't. " +
        "Try converting the video to MP4 first."
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

function blurRegion(video, box, scaleX, scaleY) {
  // Expand the detected box a little so hairlines/chins are covered too.
  const padW = box.width * BOX_PADDING;
  const padH = box.height * BOX_PADDING;
  const x = (box.originX - padW / 2) * scaleX;
  const y = (box.originY - padH / 2) * scaleY;
  const w = (box.width + padW) * scaleX;
  const h = (box.height + padH) * scaleY;

  const cx = x + w / 2;
  const cy = y + h / 2;

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, w / 2, h / 2, 0, 0, Math.PI * 2);
  ctx.clip();

  if (useCanvasBlur) {
    const radius = Math.max(12, Math.round(w / 5));
    ctx.filter = `blur(${radius}px)`;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.filter = "none";
  } else {
    // Fallback for browsers without canvas filters: chunky pixelation.
    // Sample the face region from the already-drawn frame, shrink it way
    // down, then stretch it back up with smoothing off.
    const block = Math.max(8, Math.round(w / 10));
    const tinyW = Math.max(2, Math.round(w / block));
    const tinyH = Math.max(2, Math.round(h / block));
    const tiny = session.tinyCanvas;
    tiny.width = tinyW;
    tiny.height = tinyH;
    const tctx = tiny.getContext("2d");
    tctx.drawImage(canvas, x, y, w, h, 0, 0, tinyW, tinyH);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tiny, 0, 0, tinyW, tinyH, x, y, w, h);
    ctx.imageSmoothingEnabled = true;
  }
  ctx.restore();
}

async function startProcessing() {
  if (!currentFile || !currentObjectUrl) return;
  if (!detectorReady) {
    showUploadError("The face-detection robot isn't ready yet — give it a second and try again.");
    return;
  }

  const mimeType = pickMimeType();
  if (!mimeType) {
    showUploadError(
      "Your browser doesn't support in-browser video recording (MediaRecorder). " +
        "Try a recent version of Chrome, Edge, Firefox, or Safari."
    );
    return;
  }

  clearUploadError();

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
    stickyBoxes: [],
    stickyAge: 0,
    chunks: [],
    recorder: null,
    audioCtx: null,
    mimeType,
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

    // Process the first frame (detect + blur) BEFORE recording starts, so
    // not even one unblurred frame can slip into the output.
    if (video.readyState < 2) {
      await new Promise((resolve) => {
        video.oncanplay = resolve;
        setTimeout(resolve, 3000);
      });
    }
    processFrame(s);

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
      videoBitsPerSecond: 8_000_000,
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

    // --- per-frame loop
    const onFrame = () => {
      if (!s.active) return;
      processFrame(s);
      scheduleFrame();
    };
    const scheduleFrame = () => {
      if (!s.active) return;
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
      else requestAnimationFrame(onFrame);
    };
    scheduleFrame();

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

function processFrame(s) {
  const video = s.video;

  // 1. detect
  let boxes = [];
  try {
    const result = detector.detectForVideo(video, performance.now());
    boxes = (result.detections || []).map((d) => d.boundingBox).filter(Boolean);
  } catch (err) {
    // A single bad frame shouldn't kill the run.
    console.warn("Detection hiccup:", err);
  }

  // Keep blurring the last known spots for a few frames when detection
  // momentarily loses a face (fast motion, odd angles).
  if (boxes.length > 0) {
    s.stickyBoxes = boxes;
    s.stickyAge = 0;
  } else if (s.stickyBoxes.length > 0 && s.stickyAge < BOX_STICKY_FRAMES) {
    boxes = s.stickyBoxes;
    s.stickyAge++;
  }

  // 2. draw + blur
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / video.videoWidth;
  const scaleY = canvas.height / video.videoHeight;
  for (const box of boxes) blurRegion(video, box, scaleX, scaleY);
  if (boxes.length > 0) s.faceFrames += boxes.length;

  // 3. progress
  updateProgress(video.currentTime, s.duration, s.faceFrames);
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
    (err && err.message ? err.message + " " : "Something went wrong while processing. ") +
      "No harm done — nothing left your device. Try a different file or a smaller video."
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
// Guestbook (Formspree — no storage anywhere, just an email)
// ------------------------------------------------------------------

const gbForm = $("guestbook-form");
const gbStatus = $("gb-status");

gbForm.action = `https://formspree.io/f/${FORMSPREE_ID}`;

gbForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = new FormData(gbForm);

  if (FORMSPREE_ID === "YOUR_FORM_ID") {
    // Formspree not configured yet — fall back to the visitor's mail app.
    const subject = encodeURIComponent("FaceOff!! guestbook entry");
    const body = encodeURIComponent(
      `Name: ${data.get("name") || "(anonymous)"}\nEmail: ${data.get("email") || "(none)"}\n\n${data.get("message")}`
    );
    window.location.href = `mailto:${FALLBACK_EMAIL}?subject=${subject}&body=${body}`;
    gbStatus.textContent = "📬 Opening your email app to send your note the old-fashioned way…";
    return;
  }

  gbStatus.textContent = "📨 Sending…";
  try {
    const res = await fetch(gbForm.action, {
      method: "POST",
      body: data,
      headers: { Accept: "application/json" },
    });
    if (res.ok) {
      gbForm.reset();
      gbStatus.textContent = "🌟 Thanks!! Your note is on its way. You RULE. 🌟";
    } else {
      gbStatus.textContent = "😖 Sending failed — please try again in a bit!";
    }
  } catch {
    gbStatus.textContent = "😖 Couldn't reach the mail service — check your connection and try again!";
  }
});

// ------------------------------------------------------------------
// Decorative hit counter — computed locally from the date, no storage,
// no cookies, no tracking of any kind. Pure 1997 vibes.
// ------------------------------------------------------------------

(function fakeHitCounter() {
  const daysSince1997 = Math.floor((Date.now() - Date.UTC(1997, 0, 1)) / 86400000);
  const count = 31337 + daysSince1997 * 13 + new Date().getHours();
  $("hit-counter").textContent = String(count).padStart(6, "0");
})();

// ------------------------------------------------------------------
// Boot
// ------------------------------------------------------------------

loadDetector();
