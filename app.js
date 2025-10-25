// app.js
import { PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/+esm';
/* ===========================
   UI elements
   =========================== */
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const videoUpload = document.getElementById('videoUpload');
const btnLoad = document.getElementById('btnLoad');
const btnPause = document.getElementById('btnPause');
const btnSeekStart = document.getElementById('btnSeekStart');
const dot = document.getElementById('dot');
const label = document.getElementById('label');
const avgVal = document.getElementById('avgVal');
const abdLVal = document.getElementById('abdL');
const abdRVal = document.getElementById('abdR');
const warn = document.getElementById('warn');
const startBtn = document.getElementById('start');
const dlBtn = document.getElementById('dl');
const bar = document.getElementById('bar');
const dbg = document.getElementById('dbg');
/* ===========================
   Paciento kodas
   =========================== */
let patientCode = null;
function askPatientCode() {
  let ok = false;
  while (!ok) {
    const val = prompt('Įveskite paciento/tyrimo kodą (iki 10 skaitmenų):', '');
    if (val === null) return false;
    if (/^\d{1,10}$/.test(val)) { patientCode = val; ok = true; }
    else alert('Kodas turi būti 1–10 skaitmenų.');
  }
  return true;
}
/* ===========================
   Įrašymo parametrai
   =========================== */
const RECORD_MS = 2000;
const SAMPLE_MS = 10;
const RECORD_SAMPLES = RECORD_MS / SAMPLE_MS;
let collectedData = [];
let isCollecting = false;
let sampleIdx = 0;
let sampler = null;
/* ===========================
   MediaPipe / landmarks
   =========================== */
let pose;
const LM = {
  NOSE: 0,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28
};
/* ===========================
   Math / angles
   =========================== */
const unit = (v) => { const n = Math.hypot(v.x, v.y); return n > 1e-6 ? { x: v.x / n, y: v.y / n } : { x: 0, y: 0 }; };
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const dotp = (a, b) => a.x * b.x + a.y * b.y;
const ang = (a, b) => { const c = Math.max(-1, Math.min(1, dotp(unit(a), unit(b)))); return Math.acos(c) * 180 / Math.PI; };
/* ===========================
   Midline + abduction (fixed)
   =========================== */
function bodyMidlineFromLandmarks(L) {
  const S_mid = { x: (L.leftShoulder.x + L.rightShoulder.x) / 2, y: (L.leftShoulder.y + L.rightShoulder.y) / 2 };
  const H_mid = { x: (L.leftHip.x + L.rightHip.x) / 2, y: (L.leftHip.y + L.rightHip.y) / 2 };
  // Kryptis nuo klubų į pečius (kad kampai būtų <180°)
  let midDown = unit(sub(S_mid, H_mid));
  // Jei žmogus guli ar pasviręs – užtikrina teisingą kryptį
  if (midDown.y < 0) midDown = { x: -midDown.x, y: -midDown.y };
  return { S_mid, H_mid, midDown };
}
function abductionPerHip2D(HIP, KNEE, midDown) {
  let a = ang(sub(KNEE, HIP), midDown);
  if (a > 180) a = 360 - a;
  return a;
}
/* ===========================
   Colors
   =========================== */
const SAFE = { greenMin: 30, greenMax: 45, yellowMax: 60 };
const colAbd = (a) => a >= SAFE.greenMin && a <= SAFE.greenMax ? '#34a853'
  : a > SAFE.greenMax && a <= SAFE.yellowMax ? '#f9ab00'
  : '#ea4335';
/* ===========================
   One Euro filter
   =========================== */
class OneEuro {
  constructor({ minCutoff = 1.0, beta = 0.0, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = this.dxPrev = this.tPrev = null;
  }
  static alpha(dt, cutoff) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }
  filterVec(t, x) {
    if (this.tPrev == null) { this.tPrev = t; this.xPrev = x; this.dxPrev = 0; return x; }
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
   Pose model init
   =========================== */
const MODEL_META = {
  family: 'MediaPipe Tasks PoseLandmarker',
  variant: 'pose_landmarker_full',
  precision: 'float16',
  version: '1',
  delegate: 'GPU'
};
async function initPose() {
  const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm');
  pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
      delegate: MODEL_META.delegate
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false
  });
}
const visOK = (p) => (p.visibility ?? 1) >= 0.6;
/* ===========================
   State & loop
   =========================== */
let latest = { ok: false, angles: null, midline: null, lms: null, ts: null };
let lastVideoTime = -1;
async function loop() {
  if (!pose || !video.videoWidth) { requestAnimationFrame(loop); return; }
  const tNow = performance.now();
  if (video.currentTime !== lastVideoTime) {
    const out = await pose.detectForVideo(video, tNow);
    lastVideoTime = video.currentTime;
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
        leftShoulder: smooth[LM.LEFT_SHOULDER], rightShoulder: smooth[LM.RIGHT_SHOULDER],
        leftHip: smooth[LM.LEFT_HIP], rightHip: smooth[LM.RIGHT_HIP],
        leftKnee: smooth[LM.LEFT_KNEE], rightKnee: smooth[LM.RIGHT_KNEE]
      };
      const okVis = visOK(L.leftShoulder) && visOK(L.rightShoulder) &&
        visOK(L.leftHip) && visOK(L.rightHip) &&
        visOK(L.leftKnee) && visOK(L.rightKnee);
      if (okVis) {
        const midline = bodyMidlineFromLandmarks(L);
        const abdL = abductionPerHip2D(L.leftHip, L.leftKnee, midline.midDown);
        const abdR = abductionPerHip2D(L.rightHip, L.rightKnee, midline.midDown);
        drawOverlay(L, { abdL, abdR }, midline);
        const avg = (abdL + abdR) / 2;
        avgVal.textContent = `${avg.toFixed(1)}°`;
        abdLVal.textContent = `${abdL.toFixed(0)}°`;
        abdRVal.textContent = `${abdR.toFixed(0)}°`;
        avgVal.style.color = colAbd(avg);
        abdLVal.style.color = colAbd(abdL);
        abdRVal.style.color = colAbd(abdR);
        dot.className = 'status-dot dot-active';
        warn.textContent =
          avg > SAFE.yellowMax ? '⚠️ Per didelė abdukcija (>60°).' :
            avg < SAFE.greenMin ? '⚠️ Per maža abdukcija (<30°).' :
              avg <= SAFE.greenMax ? 'Poza gera (30–45°).' :
                'Įspėjimas: 45–60° (geltona zona).';
        latest = { ok: true, angles: { abdL, abdR, avg }, midline, lms: L, ts: tNow };
        if (isCollecting && tNow - latest.ts >= SAMPLE_MS) {
          collectedData.push({
            patientCode,
            timestamp: +tNow.toFixed(1),
            avgAngle: +avg.toFixed(5),
            leftAngle: +abdL.toFixed(5),
            rightAngle: +abdR.toFixed(5),
            midline: {
              S_mid: { x: +midline.S_mid.x.toFixed(5), y: +midline.S_mid.y.toFixed(5) },
              H_mid: { x: +midline.H_mid.x.toFixed(5), y: +midline.H_mid.y.toFixed(5) }
            },
            leftHip: { x: +L.leftHip.x.toFixed(5), y: +L.leftHip.y.toFixed(5), v: +L.leftHip.visibility.toFixed(5) },
            rightHip: { x: +L.rightHip.x.toFixed(5), y: +L.rightHip.y.toFixed(5), v: +L.rightHip.visibility.toFixed(5) },
            leftKnee: { x: +L.leftKnee.x.toFixed(5), y: +L.leftKnee.y.toFixed(5), v: +L.leftKnee.visibility.toFixed(5) },
            rightKnee: { x: +L.rightKnee.x.toFixed(5), y: +L.rightKnee.y.toFixed(5), v: +L.rightKnee.visibility.toFixed(5) }
          });
          sampleIdx++;
          bar.style.width = `${(sampleIdx / RECORD_SAMPLES) * 100}%`;
          if (sampleIdx >= RECORD_SAMPLES) {
            stopCollecting();
          }
        }
      } else {
        drawOverlay(null, {}, { S_mid: null, H_mid: null, midDown: null });
        avgVal.textContent = abdLVal.textContent = abdRVal.textContent = '–';
        warn.textContent = 'Žemas matomumas / silpna poza.';
        dot.className = 'status-dot dot-idle';
        latest.ok = false;
      }
    } else {
      drawOverlay(null, {}, { S_mid: null, H_mid: null, midDown: null });
      label.textContent = 'Poza nerasta';
      dot.className = 'status-dot dot-idle';
      latest.ok = false;
    }
  }
  requestAnimationFrame(loop);
}
/* ===========================
   Drawing overlay (updated)
   =========================== */
function drawOverlay(L, angles, midline) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!L) return;
  const toPx = (p) => ({ x: p.x * canvas.width, y: p.y * canvas.height });
  const LS = toPx(L.leftShoulder), RS = toPx(L.rightShoulder);
  const LH = toPx(L.leftHip), RH = toPx(L.rightHip);
  const LK = toPx(L.leftKnee), RK = toPx(L.rightKnee);
  const Spt = toPx(midline.S_mid);
  const Hpt = toPx(midline.H_mid);
  // --- 1️⃣ Nupiešiame visus 6 LANDMARKUS kaip raudonus taškus ---
  const redPts = [LS, RS, LH, RH, LK, RK];
  ctx.fillStyle = '#ff0000';
  for (const p of redPts) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  // --- 2️⃣ Nupiešiame žalius VEKTORIUS ---
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2;
  // Midline (pečių vidurys -> klubų vidurys)
  ctx.beginPath();
  ctx.moveTo(Spt.x, Spt.y);
  ctx.lineTo(Hpt.x, Hpt.y);
  ctx.stroke();
  // Kairė koja (kelis → klubas)
  ctx.beginPath();
  ctx.moveTo(LK.x, LK.y);
  ctx.lineTo(LH.x, LH.y);
  ctx.stroke();
  // Dešinė koja (kelis → klubas)
  ctx.beginPath();
  ctx.moveTo(RK.x, RK.y);
  ctx.lineTo(RH.x, RH.y);
  ctx.stroke();
  // --- 3️⃣ Parodome kampus ties klubais ---
  const colorL = colAbd(angles.abdL);
  const colorR = colAbd(angles.abdR);
  ctx.font = '14px system-ui, -apple-system, Segoe UI, Roboto';
  ctx.textBaseline = 'bottom';
  ctx.fillStyle = colorL;
  ctx.fillText(`${angles.abdL.toFixed(0)}°`, LH.x + 8, LH.y - 8);
  ctx.fillStyle = colorR;
  ctx.fillText(`${angles.abdR.toFixed(0)}°`, RH.x + 8, RH.y - 8);
}
/* ===========================
   Data collection
   =========================== */
function startCollecting() {
  if (!patientCode && !askPatientCode()) return;
  if (!latest.ok) { alert('Poza neaptikta – negalima pradėti įrašymo.'); return; }
  collectedData = [];
  isCollecting = true;
  sampleIdx = 0;
  bar.style.width = '0%';
  startBtn.disabled = true;
  dlBtn.disabled = true;
  warn.textContent = `⏺️ Renkami duomenys (2 s) – ${patientCode}`;
  sampler = setInterval(() => {
    // Sampling logic moved to loop for accuracy
  }, SAMPLE_MS);
}
function stopCollecting() {
  clearInterval(sampler);
  sampler = null;
  isCollecting = false;
  startBtn.disabled = false;
  dlBtn.disabled = false;
  warn.textContent = `✅ Surinkta ${collectedData.length} įrašų.`;
}
function downloadJSON() {
  if (!collectedData.length) { alert('Nėra duomenų.'); return; }
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const name = `pose_data_${patientCode || 'anon'}_${ts}.json`;
  const blob = new Blob([JSON.stringify(collectedData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}
/* ===========================
   Video handling
   =========================== */
let currentFileURL = null;
btnLoad.addEventListener('click', async () => {
  const file = videoUpload.files[0];
  if (!file) { alert('Pasirinkite vaizdo failą.'); return; }
  if (!patientCode && !askPatientCode()) return;
  if (currentFileURL) URL.revokeObjectURL(currentFileURL);
  currentFileURL = URL.createObjectURL(file);
  video.src = currentFileURL;
  label.textContent = 'Kraunama vaizdo įrašas...';
  await initPose();
  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    video.play().then(() => {
      label.textContent = 'Analizuoju...';
      lastVideoTime = -1;
      requestAnimationFrame(loop);
    });
  };
});
btnPause.addEventListener('click', () => {
  if (video.paused) { video.play(); btnPause.textContent = 'Pauzė'; }
  else { video.pause(); btnPause.textContent = 'Tęsti'; }
});
btnSeekStart.addEventListener('click', () => { video.currentTime = 0; });
startBtn.addEventListener('click', startCollecting);
dlBtn.addEventListener('click', downloadJSON);
