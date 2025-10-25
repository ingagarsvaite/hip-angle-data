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
   Patient code prompt
   =========================== */
let patientCode = null;
function askPatientCode(){
  let ok = false;
  while(!ok){
    const val = prompt('Įveskite paciento/tyrimo kodą (iki 10 skaitmenų):', '');
    if (val === null) return false;
    if (/^\d{1,10}$/.test(val)) { patientCode = val; ok = true; }
    else alert('Kodas turi būti 1–10 skaitmenų.');
  }
  return true;
}

/* ===========================
   Recording parameters
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
let pose = null;
const LM = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
};

/* ===========================
   Vector math
   =========================== */
const unit = (v)=>{ const n=Math.hypot(v.x,v.y); return n>1e-6?{x:v.x/n,y:v.y/n}:{x:0,y:0}; };
const sub  = (a,b)=>({x:a.x-b.x, y:a.y-b.y});
const dotp = (a,b)=>a.x*b.x + a.y*b.y;
const ang  = (a,b)=>{ const c=Math.max(-1,Math.min(1,dotp(unit(a),unit(b)))); return Math.acos(c)*180/Math.PI; };

/* ===========================
   Midline + abduction
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

  // body axis from hips to shoulders (works for lying body)
  const midDown = unit(sub(S_mid, H_mid));

  return { S_mid, H_mid, midDown };
}

/* ===========================
   Colors
   =========================== */
const SAFE = { greenMin:30, greenMax:45, yellowMax:60 };
const colAbd = (a)=> a>=SAFE.greenMin && a<=SAFE.greenMax ? '#34a853'
                    : a>SAFE.greenMax && a<=SAFE.yellowMax ? '#f9ab00'
                    : '#ea4335';

/* ===========================
   Tilt sensor (optional)
   =========================== */
let tiltDeg = null;
let tiltOK  = null;
let sensorsEnabled = false;
function updateTiltWarn(){
  if (tiltDeg == null) return;
  if (Math.abs(tiltDeg) > 5){
    warn.textContent = `⚠️ Telefonas pakreiptas ${tiltDeg.toFixed(1)}° (>5°). Ištiesinkite įrenginį.`;
  }
}
function onDeviceOrientation(e){
  const portrait = window.innerHeight >= window.innerWidth;
  const primaryTilt = portrait ? (e.gamma ?? 0) : (e.beta ?? 0);
  tiltDeg = Number(primaryTilt) || 0;
  tiltOK  = Math.abs(tiltDeg) <= 5;
  updateTiltWarn();
}
async function enableSensors(){
  try{
    if (typeof DeviceOrientationEvent !== 'undefined' &&
        typeof DeviceOrientationEvent.requestPermission === 'function'){
      const perm = await DeviceOrientationEvent.requestPermission();
      if (perm !== 'granted') throw new Error('Leidimas nesuteiktas');
    }
    window.addEventListener('deviceorientation', onDeviceOrientation, true);
    sensorsEnabled = true;
  }catch(e){
    console.warn('Nepavyko įjungti tilt jutiklio:', e);
  }
}

/* ===========================
   Canvas resizing
   =========================== */
function resizeCanvas(){
  const w = video.clientWidth, h = video.clientHeight;
  if (!w || !h) return;
  canvas.width = w; canvas.height = h;
}
window.addEventListener('resize', resizeCanvas);

/* ===========================
   One Euro filter
   =========================== */
class OneEuro {
  constructor({ minCutoff=1.0, beta=0.0, dCutoff=1.0 }={}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = null;
    this.tPrev = null;
  }
  static alpha(dt, cutoff){
    const tau = 1.0/(2*Math.PI*cutoff);
    return 1.0/(1.0 + tau/dt);
  }
  filterVec(t, x){
    if (this.tPrev == null){
      this.tPrev = t; this.xPrev = x; this.dxPrev = 0;
      return x;
    }
    const dt = Math.max(1e-6, (t - this.tPrev)/1000);
    const dx = (x - this.xPrev)/dt;
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
function ensureEuroFilters(){
  if (euro) return;
  euro = Array.from({length:33}, ()=>(({
    x:new OneEuro({minCutoff:2.0, beta:0.3, dCutoff:1.0}),
    y:new OneEuro({minCutoff:2.0, beta:0.3, dCutoff:1.0}),
    z:new OneEuro({minCutoff:1.0, beta:0.1, dCutoff:1.0})
  })));
}

/* ===========================
   MediaPipe Pose init
   =========================== */
const MODEL_META = {
  family: 'MediaPipe Tasks PoseLandmarker',
  variant: 'pose_landmarker_full',
  precision: 'float16',
  version: '1',
  delegate: 'GPU'
};

async function initPose(){
  if (pose) return;
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
  );
  pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions:{
      modelAssetPath:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
      delegate: MODEL_META.delegate
    },
    runningMode:'VIDEO',
    numPoses:1,
    minPoseDetectionConfidence:0.3,
    minPosePresenceConfidence:0.3,
    minTrackingConfidence:0.3,
    outputSegmentationMasks:false
  });
}

/* ===========================
   Utility & Drawing
   =========================== */
const visOK = (p)=> (p.visibility ?? 1) >= 0.2;

function skeletonOK(L){
  const shoulderDist = Math.hypot(L.leftShoulder.x - L.rightShoulder.x, L.leftShoulder.y - L.rightShoulder.y);
  const hipDist = Math.hypot(L.leftHip.x - L.rightHip.x, L.leftHip.y - L.rightHip.y);
  const torsoLen = Math.hypot(
    (L.leftShoulder.x+L.rightShoulder.x)/2 - (L.leftHip.x+L.rightHip.x)/2,
    (L.leftShoulder.y+L.rightShoulder.y)/2 - (L.leftHip.y+L.rightHip.y)/2
  );
  if (shoulderDist < 0.02 || hipDist < 0.02) return false;
  if (torsoLen < 0.02) return false;
  if (!Number.isFinite(shoulderDist + hipDist + torsoLen)) return false;
  return true;
}

function drawOverlay(L, angles, midline){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!L) return;

  const toPx = (p)=>({ x:p.x*canvas.width, y:p.y*canvas.height });
  const LS = toPx(L.leftShoulder),  RS = toPx(L.rightShoulder);
  const LH = toPx(L.leftHip),       RH = toPx(L.rightHip);
  const LK = toPx(L.leftKnee),      RK = toPx(L.rightKnee);
  const Spt = toPx(midline.S_mid);
  const Hpt = toPx(midline.H_mid);

  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(Spt.x, Spt.y); ctx.lineTo(Hpt.x, Hpt.y); ctx.stroke();

  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 5; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(LH.x,LH.y); ctx.lineTo(LK.x,LK.y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(RH.x,RH.y); ctx.lineTo(RK.x,RK.y); ctx.stroke();

  ctx.fillStyle = '#ffffff';
  for (const p of [LS, RS, LH, RH, LK, RK]){
    ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fill();
  }

  const colorL = colAbd(angles.abdL);
  const colorR = colAbd(angles.abdR);
  ctx.font = '14px system-ui, -apple-system, Segoe UI, Roboto';
  ctx.textBaseline = 'bottom';

  function labelAt(pt, text, col){
    const pad = 4, offY = -8;
    const m = ctx.measureText(text);
    const w = m.width + pad*2, h = 18;
    const x = Math.min(Math.max(0, pt.x - w/2), canvas.width - w);
    const y = Math.min(Math.max(h, pt.y + offY), canvas.height);
    ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(x, y-h, w, h);
    ctx.fillStyle = col; ctx.fillText(text, x + pad, y - 4);
  }
  if (isFinite(angles.abdL)) labelAt(LH, `${angles.abdL.toFixed(0)}°`, colorL);
  if (isFinite(angles.abdR)) labelAt(RH, `${angles.abdR.toFixed(0)}°`, colorR);
}

/* ===========================
   Main loop
   =========================== */
let rafHandle = null;
let lastVideoTime = -1;
let latest = { ok:false, angles:null, midline:null, lms:null, ts:null };

async function loop(){
  if (!pose || !video.videoWidth) {
    rafHandle = requestAnimationFrame(loop);
    return;
  }

  const tNow = performance.now();

  if (video.currentTime !== lastVideoTime){
    const out = await pose.detectForVideo(video, tNow);
    lastVideoTime = video.currentTime;

    if (out.landmarks && out.landmarks.length>0){
      ensureEuroFilters();
      const raw = out.landmarks[0];
      const smooth = raw.map((p,i)=>(({
        x: euro[i].x.filterVec(tNow, p.x),
        y: euro[i].y.filterVec(tNow, p.y),
        z: euro[i].z.filterVec(tNow, p.z ?? 0),
        visibility: p.visibility ?? 1
      })));

      const L = {
        leftShoulder: smooth[LM.LEFT_SHOULDER],  rightShoulder: smooth[LM.RIGHT_SHOULDER],
        leftHip:      smooth[LM.LEFT_HIP],       rightHip:      smooth[LM.RIGHT_HIP],
        leftKnee:     smooth[LM.LEFT_KNEE],      rightKnee:     smooth[LM.RIGHT_KNEE]
      };

      const okVis = visOK(L.leftShoulder) && visOK(L.rightShoulder) &&
                    visOK(L.leftHip) && visOK(L.rightHip) &&
                    visOK(L.leftKnee) && visOK(L.rightKnee);

      if (okVis && skeletonOK(L)){
        const midline = bodyMidlineFromLandmarks(L);
        const abdL = ang(sub(L.leftKnee,  L.leftHip),  midline.midDown);
        const abdR = ang(sub(L.rightKnee, L.rightHip), midline.midDown);
        const fix = (a)=> (a > 180 ? 360 - a : a);
        const abdL_fixed = fix(abdL);
        const abdR_fixed = fix(abdR);

        drawOverlay(L, {abdL:abdL_fixed, abdR:abdR_fixed}, midline);

        const avg = (abdL_fixed + abdR_fixed)/2;
        avgVal.textContent = isFinite(avg)? `${avg.toFixed(1)}°` : '–';
        abdLVal.textContent = isFinite(abdL_fixed)? `${abdL_fixed.toFixed(0)}°` : '–';
        abdRVal.textContent = isFinite(abdR_fixed)? `${abdR_fixed.toFixed(0)}°` : '–';

        avgVal.style.color = colAbd(avg);
        abdLVal.style.color = colAbd(abdL_fixed);
        abdRVal.style.color = colAbd(abdR_fixed);
        dot.className = 'status-dot dot-active';

        if (avg > SAFE.yellowMax)       warn.textContent = '⚠️ Per didelė abdukcija (>60°).';
        else if (avg < SAFE.greenMin)   warn.textContent = '⚠️ Per maža abdukcija (<30°).';
        else if (avg <= SAFE.greenMax)  warn.textContent = 'Poza gera (30–45°).';
        else                             warn.textContent = 'Įspėjimas: 45–60° (geltona zona).';

        latest.ok = true;
        latest.angles = { abdL:abdL_fixed, abdR:abdR_fixed, avg };
        latest.midline = midline;
        latest.lms = L;
        latest.ts = tNow;
      } else {
        drawOverlay(null,{}, {S_mid:null,H_mid:null,midDown:null});
        avgVal.textContent='–'; abdLVal.textContent='–'; abdRVal.textContent='–';
        avgVal.style.color='#e5e7eb';
        abdLVal.style.color='#e5e7eb';
        abdRVal.style.color='#e5e7eb';
        dot.className = 'status-dot dot-idle';
        warn.textContent = 'Žemas matomumas / silpna poza – pataisykite vaizdą arba apšvietimą.';
        latest.ok = false;
        latest.ts = tNow;
      }
    }
  }

  rafHandle = requestAnimationFrame(loop);
}

/* ===========================
   Video handling
   =========================== */
let currentFileURL = null;
btnLoad.addEventListener('click', async ()=>{
  const file = videoUpload.files[0];
  if (!file){ alert('Pasirinkite vaizdo failą.'); return; }

  if (!patientCode){
    const cont = askPatientCode();
    if (!cont) return;
  }

  if (currentFileURL){ URL.revokeObjectURL(currentFileURL); }
  currentFileURL = URL.createObjectURL(file);
  video.src = currentFileURL;
  video.playsInline = true;
  video.muted = true;
  video.pause();
  label.textContent = 'Kraunama vaizdo įrašas…';
  resizeCanvas();

  try{
    await initPose();
    video.onloadedmetadata = () => {
      resizeCanvas();
      video.play().then(()=>{
        label.textContent = 'Vaizdas groja';
        startBtn.disabled = false;
        if (rafHandle) cancelAnimationFrame(rafHandle);
        lastVideoTime = -1;
        rafHandle = requestAnimationFrame(loop);
      });
    };
  }catch(err){
    console.error('Pose init error:', err);
    alert('Nepavyko įkelti modelio.');
  }
});

btnPause.addEventListener('click', ()=>{
  if (!video.src) return;
  if (video.paused) { video.play(); btnPause.textContent = 'Pauzė'; }
  else { video.pause(); btnPause.textContent = 'Tęsti'; }
});

btnSeekStart.addEventListener('click', ()=>{
  if (!video

