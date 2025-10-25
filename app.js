// app.js
import { PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/+esm';

/* ===========================
   UI ELEMENTS
   =========================== */
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const videoUpload = document.getElementById('videoUpload');
const btnLoad = document.getElementById('btnLoad');
const btnPause = document.getElementById('btnPause');
const btnSeekStart = document.getElementById('btnSeekStart');
const label = document.getElementById('label');
const warn = document.getElementById('warn');

/* ===========================
   BASIC VECTOR MATH
   =========================== */
const unit = (v) => {
  const n = Math.hypot(v.x, v.y);
  return n > 1e-6 ? { x: v.x / n, y: v.y / n } : { x: 0, y: 0 };
};
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const dotp = (a, b) => a.x * b.x + a.y * b.y;
const ang = (a, b) => {
  const c = Math.max(-1, Math.min(1, dotp(unit(a), unit(b))));
  return Math.acos(c) * 180 / Math.PI;
};

/* ===========================
   LANDMARK INDEXES
   =========================== */
const LM = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26
};

/* ===========================
   MIDLINE CALCULATION
   =========================== */
function bodyMidlineFromLandmarks(L) {
  const S_mid = {
    x: (L.leftShoulder.x + L.rightShoulder.x) / 2,
    y: (L.leftShoulder.y + L.rightShoulder.y) / 2
  };
  const H_mid = {
    x: (L.leftHip.x + L.rightHip.x) / 2,
    y: (L.leftHip.y + L.rightHip.y) / 2
  };
  const midDown = unit(sub(H_mid, S_mid)); // from shoulders to hips
  return { S_mid, H_mid, midDown };
}

/* ===========================
   ONE EURO FILTER
   =========================== */
class OneEuro {
  constructor({ minCutoff = 2.0, beta = 0.3, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = null;
    this.tPrev = null;
  }
  static alpha(dt, cutoff) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }
  filterVec(t, x) {
    if (this.tPrev == null) {
      this.tPrev = t; this.xPrev = x; this.dxPrev = 0;
      return x;
    }
    const dt = Math.max(1e-6, (t - this.tPrev) / 1000);
    const dx = (x - this.xPrev) / dt;
    const aD = OneEuro.alpha(dt, this.dCutoff);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const aX = OneEuro.alpha(dt, cutoff);
    const xHat = aX * x + (1 - aX) * this.xPrev;
    this.tPrev = t; this.xPrev = xHat; this.dxPrev = dxHat;
    return xHat;
  }
}

let euro = null;
function ensureEuroFilters() {
  if (euro) return;
  euro = Array.from({ length: 33 }, () => ({
    x: new OneEuro({ minCutoff: 2.0, beta: 0.3, dCutoff: 1.0 }),
    y: new OneEuro({ minCutoff: 2.0, beta: 0.3, dCutoff: 1.0 }),
    z: new OneEuro({ minCutoff: 1.0, beta: 0.1, dCutoff: 1.0 })
  }));
}

/* ===========================
   MEDIAPIPE INIT
   =========================== */
let pose = null;
async function initPose() {
  if (pose) return;
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
  );
  pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
      delegate: 'AUTO'
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.3,
    minPosePresenceConfidence: 0.3,
    minTrackingConfidence: 0.3
  });
}

/* ===========================
   DRAWING FUNCTION
   =========================== */
function drawOverlay(L, midline) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!L) return;

  const toPx = (p) => ({ x: p.x * canvas.width, y: p.y * canvas.height });

  const LS = toPx(L.leftShoulder), RS = toPx(L.rightShoulder);
  const LH = toPx(L.leftHip), RH = toPx(L.rightHip);
  const LK = toPx(L.leftKnee), RK = toPx(L.rightKnee);
  const Spt = toPx(midline.S_mid), Hpt = toPx(midline.H_mid);

  // midline (green)
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(Spt.x, Spt.y);
  ctx.lineTo(Hpt.x, Hpt.y);
  ctx.stroke();

  // legs (green)
  ctx.beginPath();
  ctx.moveTo(LH.x, LH.y);
  ctx.lineTo(LK.x, LK.y);
  ctx.moveTo(RH.x, RH.y);
  ctx.lineTo(RK.x, RK.y);
  ctx.stroke();

  // joints (red)
  ctx.fillStyle = '#ff0000';
  for (const p of [LS, RS, LH, RH, LK, RK]) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

/* ===========================
   MAIN VIDEO PROCESS LOOP
   =========================== */
async function processFrame(now, metadata) {
  if (!pose) return;
  const tNow = performance.now();
  const out = await pose.detectForVideo(video, tNow);
  if (out.landmarks && out.landmarks.length > 0) {
    ensureEuroFilters();
    const raw = out.landmarks[0];
    const smooth = raw.map((p, i) => ({
      x: euro[i].x.filterVec(tNow, p.x),
      y: euro[i].y.filterVec(tNow, p.y),
      z: euro[i].z.filterVec(tNow, p.z ?? 0),
      visibility: p.visibility ?? 1
    }));
    const L = {
      leftShoulder: smooth[LM.LEFT_SHOULDER],
      rightShoulder: smooth[LM.RIGHT_SHOULDER],
      leftHip: smooth[LM.LEFT_HIP],
      rightHip: smooth[LM.RIGHT_HIP],
      leftKnee: smooth[LM.LEFT_KNEE],
      rightKnee: smooth[LM.RIGHT_KNEE]
    };
    const midline = bodyMidlineFromLandmarks(L);
    drawOverlay(L, midline);
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  video.requestVideoFrameCallback(processFrame);
}

/* ===========================
   VIDEO CONTROLS
   =========================== */
let currentFileURL = null;
btnLoad.addEventListener('click', async () => {
  const file = videoUpload.files[0];
  if (!file) {
    alert('Pasirinkite vaizdo failą.');
    return;
  }

  if (currentFileURL) URL.revokeObjectURL(currentFileURL);
  currentFileURL = URL.createObjectURL(file);
  video.src = currentFileURL;
  video.load();
  video.pause();
  label.textContent = 'Kraunama vaizdo įrašas...';

  await initPose();

  video.onloadeddata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    label.textContent = 'Analizuoju vaizdą...';
    video.play();
    console.log('Modelis įkeltas, analizuoju...');
    video.requestVideoFrameCallback(processFrame);
  };
});

btnPause.addEventListener('click', () => {
  if (video.paused) {
    video.play();
    btnPause.textContent = 'Pauzė';
  } else {
    video.pause();
    btnPause.textContent = 'Tęsti';
  }
});

btnSeekStart.addEventListener('click', () => {
  video.currentTime = 0;
});
