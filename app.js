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
   Patient code prompt (unchanged)
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
   Recording parameters (same)
   =========================== */
const RECORD_MS = 2000;                 // 2 s
const SAMPLE_MS = 10;                   // 10 ms
const RECORD_SAMPLES = RECORD_MS / SAMPLE_MS; // 200

let collectedData = [];
let isCollecting = false;
let sampleIdx = 0;
let sampler = null;

/* ===========================
   MediaPipe / landmarks
   =========================== */
let pose = null; // PoseLandmarker instance
const LM = {
  NOSE: 0,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28
};

/* ===========================
   Vector math / angles
   =========================== */
const unit = (v)=>{ const n=Math.hypot(v.x,v.y); return n>1e-6?{x:v.x/n,y:v.y/n}:{x:0,y:0}; };
const sub  = (a,b)=>({x:a.x-b.x, y:a.y-b.y});
const dotp = (a,b)=>a.x*b.x + a.y*b.y;
const ang  = (a,b)=>{ const c=Math.max(-1,Math.min(1, dotp(unit(a),unit(b)))); return Math.acos(c)*180/Math.PI; };

/* ===========================
   Midline + abduction
   =========================== */
function bodyMidlineFromLandmarks(L){
  const S_mid = { x:(L.leftShoulder.x + L.rightShoulder.x)/2, y:(L.leftShoulder.y + L.rightShoulder.y)/2 };
  const H_mid = { x:(L.leftHip.x + L.rightHip.x)/2,           y:(L.leftHip.y + L.rightHip.y)/2 };
  let midDown = unit(sub(H_mid, S_mid));
  if (midDown.y < 0) midDown = { x:-midDown.x, y:-midDown.y };
  return { S_mid, H_mid, midDown };
}
function abductionPerHip2D(HIP, KNEE, midDown){
  return ang(sub(KNEE, HIP), midDown);
}

/* ===========================
   Colors
   =========================== */
const SAFE = { greenMin:30, greenMax:45, yellowMax:60 };
const colAbd = (a)=> a>=SAFE.greenMin && a<=SAFE.greenMax ? '#34a853'
                    : a>SAFE.greenMax && a<=SAFE.yellowMax ? '#f9ab00'
                    : '#ea4335';

/* ===========================
   Tilt sensor (kept but optional)
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
   One Euro filter (per-landmark, per-axis)
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
  filterVec(t, x){ // x is number
    if (this.tPrev == null){
      this.tPrev = t; this.xPrev = x; this.dxPrev = 0;
      return x;
    }
    const dt = Math.max(1e-6, (t - this.tPrev)/1000); // t in ms → s
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
    minPoseDetectionConfidence:0.5,
    minPosePresenceConfidence:0.5,
    minTrackingConfidence:0.5,
    outputSegmentationMasks:false
  });
}

/* ===========================
   Utility & Drawing code (unchanged)
   =========================== */
const visOK = (p)=> (p.visibility ?? 1) >= 0.6;
let latest = { ok:false, angles:null, midline:null, lms:null, ts:null };
let startRecordTs = null;

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

function updateDebug(L){
  if (!L){ dbg.textContent = 'Laukiu pozos…'; return; }
  const fmt = (p)=>`(${p.x.toFixed(3)}, ${p.y.toFixed(3)}, v=${(p.visibility??1).toFixed(2)})`;
  dbg.textContent =
    `LS ${fmt(L.leftShoulder)}  RS ${fmt(L.rightShoulder)}\n`+
    `LH ${fmt(L.leftHip)}  RH ${fmt(L.rightHip)}\n`+
    `LK ${fmt(L.leftKnee)} RK ${fmt(L.rightKnee)}`;
}

function skeletonOK(L){
  const shoulderDist = Math.hypot(L.leftShoulder.x - L.rightShoulder.x, L.leftShoulder.y - L.rightShoulder.y);
  const hipDist = Math.hypot(L.leftHip.x - L.rightHip.x, L.leftHip.y - L.rightHip.y);
  const torsoLen = Math.hypot(
    (L.leftShoulder.x+L.rightShoulder.x)/2 - (L.leftHip.x+L.rightHip.x)/2,
    (L.leftShoulder.y+L.rightShoulder.y)/2 - (L.leftHip.y+L.rightHip.y)/2
  );
  if (shoulderDist < 0.05 || hipDist < 0.05) return false;
  if (torsoLen < 0.08) return false;
  if (!Number.isFinite(shoulderDist + hipDist + torsoLen)) return false;
  return true;
}

/* ===========================
   Main loop for processing video frames
   =========================== */
let rafHandle = null;
let lastVideoTime = -1;

async function loop(){
  if (!pose || !video.videoWidth) {
    rafHandle = requestAnimationFrame(loop);
    return;
  }

  const tNow = performance.now();
  // Only run detection when the video has advanced to a new frame/time.
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

      updateDebug(L);

      const okVis = visOK(L.leftShoulder) && visOK(L.rightShoulder) &&
                    visOK(L.leftHip) && visOK(L.rightHip) &&
                    visOK(L.leftKnee) && visOK(L.rightKnee);

      if (okVis && skeletonOK(L)){
        const midline = bodyMidlineFromLandmarks(L);
        const abdL = abductionPerHip2D(L.leftHip,  L.leftKnee,  midline.midDown);
        const abdR = abductionPerHip2D(L.rightHip, L.rightKnee, midline.midDown);

        drawOverlay(L, {abdL, abdR}, midline);

        const avg = (abdL + abdR)/2;
        avgVal.textContent = isFinite(avg)? `${avg.toFixed(1)}°` : '–';
        abdLVal.textContent = isFinite(abdL)? `${abdL.toFixed(0)}°` : '–';
        abdRVal.textContent = isFinite(abdR)? `${abdR.toFixed(0)}°` : '–';

        avgVal.style.color = colAbd(avg);
        abdLVal.style.color = colAbd(abdL);
        abdRVal.style.color = colAbd(abdR);
        dot.className = 'status-dot dot-active';

        if (avg > SAFE.yellowMax)       warn.textContent = '⚠️ Per didelė abdukcija (>60°).';
        else if (avg < SAFE.greenMin)   warn.textContent = '⚠️ Per maža abdukcija (<30°).';
        else if (avg <= SAFE.greenMax)  warn.textContent = 'Poza gera (30–45°).';
        else                             warn.textContent = 'Įspėjimas: 45–60° (geltona zona).';

        latest.ok = true;
        latest.angles = { abdL, abdR, avg };
        latest.midline = midline;
        latest.lms = L;
        latest.ts = tNow;

      } else {
        drawOverlay(null,{}, {S_mid:null,H_mid:null,midDown:null});
        avgVal.textContent='–'; abdLVal.textContent='–'; abdRVal.textContent='–';
        avgVal.style.color='#e5e7eb'; abdLVal.style.color='#e5e7eb'; abdRVal.style.color='#e5e7eb';
        dot.className = 'status-dot dot-idle';
        warn.textContent = 'Žemas matomumas / silpna poza – pataisykite vaizdą arba apšvietimą.';
        latest.ok = false;
        latest.ts = tNow;
      }
    } else {
      drawOverlay(null,{}, {S_mid:null,H_mid:null,midDown:null});
      label.textContent = 'Poza nerasta';
      dot.className = 'status-dot dot-idle';
      latest.ok = false;
      latest.ts = tNow;
    }
  }

  rafHandle = requestAnimationFrame(loop);
}

/* ===========================
   Recording (time-aligned) - unchanged except using latest.ts
   =========================== */
function startCollect(){
  if (isCollecting) return;
  if (!patientCode){
    const cont = askPatientCode();
    if (!cont) return;
  }

  isCollecting = true;
  collectedData = [];
  sampleIdx = 0;
  bar.style.width = '0%';
  dot.className = 'status-dot dot-rec';
  label.textContent = 'Įrašau (2 s)…';
  dlBtn.disabled = true;

  // Tilt guard (for mobile playback if user still cares)
  if (tiltDeg != null && Math.abs(tiltDeg) > 5){
    const proceed = confirm(`Telefonas pakreiptas ${tiltDeg.toFixed(1)}° (>5°).\nAr tikrai norite tęsti įrašą?`);
    if (!proceed){ stopCollect(); return; }
  }

  startRecordTs = performance.now();
  sampler = setInterval(()=>{
    const now = performance.now();
    const prog = ((sampleIdx+1)/RECORD_SAMPLES)*100;
    bar.style.width = `${Math.min(100,prog)}%`;

    if (sampleIdx >= RECORD_SAMPLES){
      stopCollect();
      return;
    }

    const timeSec = +(((sampleIdx+1)*SAMPLE_MS)/1000).toFixed(2);
    const sourceTs = latest.ts ?? now;

    if (latest.ok && latest.angles && latest.midline && latest.lms){
      const L = latest.lms, md = latest.midline, A = latest.angles;
      collectedData.push({
        timestamp: Date.now(),
        sourceTimestampMs: Math.round(sourceTs),
        time: timeSec,
        patientCode: patientCode || null,
        model: { ...MODEL_META },
        angles: {
          abductionLeft:  +A.abdL.toFixed(2),
          abductionRight: +A.abdR.toFixed(2),
          average:        +A.avg.toFixed(2)
        },
        device: {
          tiltDeg: tiltDeg == null ? null : +tiltDeg.toFixed(2),
          tiltOK: tiltDeg == null ? null : (Math.abs(tiltDeg) <= 5)
        },
        midline: {
          from: { x:+md.S_mid.x.toFixed(4), y:+md.S_mid.y.toFixed(4) },
          to:   { x:+md.H_mid.x.toFixed(4), y:+md.H_mid.y.toFixed(4) }
        },
        midlineOffset: { dx: 0, dy: 0 },
        landmarks: {
          leftShoulder:  { x:L.leftShoulder.x,  y:L.leftShoulder.y,  v:L.leftShoulder.visibility??1 },
          rightShoulder: { x:L.rightShoulder.x, y:L.rightShoulder.y, v:L.rightShoulder.visibility??1 },
          leftHip:       { x:L.leftHip.x,       y:L.leftHip.y,       v:L.leftHip.visibility??1 },
          rightHip:      { x:L.rightHip.x,      y:L.rightHip.y,      v:L.rightHip.visibility??1 },
          leftKnee:      { x:L.leftKnee.x,      y:L.leftKnee.y,      v:L.leftKnee.visibility??1 },
          rightKnee:     { x:L.rightKnee.x,     y:L.rightKnee.y,     v:L.rightKnee.visibility??1 }
        }
      });
    } else {
      collectedData.push({
        timestamp: Date.now(),
        sourceTimestampMs: Math.round(sourceTs),
        time: timeSec,
        patientCode: patientCode || null,
        model: { ...MODEL_META },
        angles: { abductionLeft:null, abductionRight:null, average:null },
        device: {
          tiltDeg: tiltDeg == null ? null : +tiltDeg.toFixed(2),
          tiltOK: tiltDeg == null ? null : (Math.abs(tiltDeg) <= 5)
        },
        midline: null,
        midlineOffset: { dx:0, dy:0 },
        landmarks: null,
        note: 'pose_not_confident'
      });
    }

    sampleIdx++;
  }, SAMPLE_MS);
}

function stopCollect(){
  if (sampler){ clearInterval(sampler); sampler = null; }
  isCollecting = false;
  dot.className = 'status-dot dot-active';
  label.textContent = collectedData.length ? `Išsaugota ${collectedData.length} mėginių` : 'Įrašas be mėginių';
  dlBtn.disabled = collectedData.length===0;
}

/* ===========================
   Download JSON (same)
   =========================== */
function downloadJSON(){
  if (!collectedData.length) return;
  const blob = new Blob([JSON.stringify(collectedData,null,2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `pose_data_${new Date().toISOString().replace(/:/g,'-')}.json`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ===========================
   Video file handling
   =========================== */
let currentFileURL = null;

btnLoad.addEventListener('click', async ()=>{
  const file = videoUpload.files[0];
  if (!file){
    alert('Pasirinkite vaizdo failą (MP4/WebM).');
    return;
  }

  // Optional: ask for patient code at load time
  if (!patientCode){
    const cont = askPatientCode();
    if (!cont) return;
  }

  // Revoke previous URL
  if (currentFileURL){
    URL.revokeObjectURL(currentFileURL);
    currentFileURL = null;
  }

  currentFileURL = URL.createObjectURL(file);
  video.src = currentFileURL;
  video.playsInline = true;
  video.muted = true;
  video.pause(); // ensure event sequence
  label.textContent = 'Kraunama vaizdo įrašas…';
  resizeCanvas();

  // Initialize pose model if needed, then play and start loop
  try{
    await initPose();
    // Optional: enable device orientation sensor if user wants tilt warnings on mobile
    // do not call enableSensors() automatically to respect privacy; user can grant separately if desired
    video.onloadedmetadata = () => {
      resizeCanvas();
      video.play().then(()=>{
        label.textContent = 'Vaizdas groja';
        startBtn.disabled = false;
        // Start the detection loop
        if (rafHandle) cancelAnimationFrame(rafHandle);
        lastVideoTime = -1;
        rafHandle = requestAnimationFrame(loop);
      }).catch(err=>{
        console.error('Vaizdo paleidimo klaida', err);
        label.textContent = 'Klaida paleidžiant vaizdą';
      });
    };
  }catch(err){
    console.error('Nepavyko inicializuoti Pose modelio:', err);
    alert('Nepavyko įkelti modelio. Patikrinkite interneto ryšį ir bandykite dar kartą.');
  }
});

btnPause.addEventListener('click', ()=>{
  if (!video.src) return;
  if (video.paused) { video.play(); btnPause.textContent = 'Pauzė'; }
  else { video.pause(); btnPause.textContent = 'Tęsti'; }
});

btnSeekStart.addEventListener('click', ()=>{
  if (!video.src) return;
  try { video.currentTime = 0; } catch(e){ console.warn(e); }
});

/* ===========================
   Events wiring
   =========================== */
startBtn.addEventListener('click', startCollect);
dlBtn.addEventListener('click', downloadJSON);

// When page loads, give instructions in UI
document.addEventListener('DOMContentLoaded', ()=>{
  label.textContent = 'Pasirinkite vaizdo failą ir paspauskite "Įkelti ir paleisti".';
  startBtn.disabled = true;
  dlBtn.disabled = true;
  resizeCanvas();
});
