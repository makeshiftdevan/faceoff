/* FaceOff detection worker: the entire YuNet pipeline runs here, off the
 * main thread — downscale, pixel readback, tensor build, ONNX inference,
 * output decode, and NMS. The main thread only transfers an ImageBitmap in
 * and gets boxes (in the bitmap's coordinates) back. */

import * as ort from "./vendor/ort/ort.wasm.bundle.min.mjs";

let session = null;
let detectCanvas = null;

// Pre/post-processing follows OpenCV's FaceDetectorYN
// (modules/objdetect/src/face_detect.cpp): BGR float input at native scale,
// per-stride grids where score = sqrt(cls * obj), box center = (cell +
// offset) * stride, size = exp(regression) * stride, then greedy IoU NMS.
function iou(a, b) {
  const x1 = Math.max(a.originX, b.originX);
  const y1 = Math.max(a.originY, b.originY);
  const x2 = Math.min(a.originX + a.width, b.originX + b.width);
  const y2 = Math.min(a.originY + a.height, b.originY + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  return inter / (a.width * a.height + b.width * b.height - inter);
}

async function detect(bitmap, maxSide, scoreThreshold, nmsIou) {
  const sw = bitmap.width;
  const sh = bitmap.height;
  const scale = Math.min(1, maxSide / Math.max(sw, sh));
  const dw0 = Math.max(1, Math.round(sw * scale));
  const dh0 = Math.max(1, Math.round(sh * scale));
  const dw = Math.ceil(dw0 / 32) * 32;
  const dh = Math.ceil(dh0 / 32) * 32;

  if (!detectCanvas || detectCanvas.width !== dw || detectCanvas.height !== dh) {
    detectCanvas = new OffscreenCanvas(dw, dh);
  }
  const dctx = detectCanvas.getContext("2d", { willReadFrequently: true });
  dctx.fillStyle = "#000";
  dctx.fillRect(0, 0, dw, dh);
  dctx.drawImage(bitmap, 0, 0, sw, sh, 0, 0, dw0, dh0);
  bitmap.close();

  const rgba = dctx.getImageData(0, 0, dw, dh).data;
  const n = dw * dh;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i] = rgba[i * 4 + 2];         // B
    input[n + i] = rgba[i * 4 + 1];     // G
    input[2 * n + i] = rgba[i * 4];     // R
  }
  const feeds = {};
  feeds[session.inputNames[0]] = new ort.Tensor("float32", input, [1, 3, dh, dw]);
  const out = await session.run(feeds);

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
        if (score < scoreThreshold) continue;
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
    if (kept.every((k) => iou(k, box) <= nmsIou)) kept.push(box);
  }

  const back = sw / dw0; // undo the downscale
  return kept.map((b) => ({
    score: b.score,
    originX: b.originX * back,
    originY: b.originY * back,
    width: b.width * back,
    height: b.height * back,
  }));
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      ort.env.wasm.wasmPaths = msg.wasmPaths;
      session = await ort.InferenceSession.create(msg.modelUrl, {
        executionProviders: ["wasm"],
      });
      // Warm up so the first real frame doesn't pay the compile cost.
      const warm = new OffscreenCanvas(96, 96);
      warm.getContext("2d").fillRect(0, 0, 96, 96);
      const bmp = await createImageBitmap(warm);
      await detect(bmp, 96, 0.5, 0.3);
      postMessage({ type: "ready" });
    } else if (msg.type === "detect") {
      const boxes = await detect(msg.bitmap, msg.maxSide, msg.score, msg.nmsIou);
      postMessage({ type: "result", id: msg.id, boxes });
    }
  } catch (err) {
    postMessage({ type: "error", id: msg.id, message: String((err && err.message) || err) });
  }
};
