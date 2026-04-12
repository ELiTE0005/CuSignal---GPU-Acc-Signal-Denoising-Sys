/**
 * CuSignal GPU Signal Processing Simulator
 * =========================================
 * Simulates the full RadarScenes + Sonar pipeline from your roadmap:
 *   Phase 1 — Data loading & parsing
 *   Phase 2 — Real-time FFT / beamforming / CFAR
 *   Phase 3 — Multi-object tracking (Kalman + Hungarian)
 *
 * To plug in real RadarScenes data, see the bottom of this file:
 * the `injectRealFrame()` function accepts a parsed HDF5 frame object.
 */

'use strict';

// ── Detect dark mode ────────────────────────────────────────────────────────
const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

const THEME = {
  bg:      isDark ? '#111110' : '#ffffff',
  grid:    isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
  text:    isDark ? '#c2c0b6' : '#3d3d3a',
  sweep:   'rgba(0,220,100,0.16)',
  types:   ['#4CAF8A', '#5B9BD5', '#E67E3A', '#A86CC1'],
  names:   ['Pedestrian', 'Car', 'Truck', 'Motorcycle'],
  sonar:   '#D4585A',
  clutter: isDark ? 'rgba(180,178,169,0.3)' : 'rgba(95,94,90,0.28)',
};

// ── Utility ──────────────────────────────────────────────────────────────────
function rand(lo, hi) { return lo + Math.random() * (hi - lo); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Simulation State ──────────────────────────────────────────────────────────
const STATE = {
  running:     true,
  phase:       1,         // 1 | 2 | 3
  mode:        'radar',   // 'radar' | 'sonar' | 'fused'
  speed:       2,
  noiseLevel:  25,
  numTargets:  6,
  frame:       0,
  sweepAngle:  0,
  targets:     [],
  clutter:     [],
  sonarTargets:[],
  trackHistory:{},
  nextId:      0,
  fps:         0,
  fpsFrames:   0,
  fpsLastTs:   performance.now(),
  // For real data injection (set to true to use injectRealFrame)
  useRealData: false,
};

// ── Target factory ─────────────────────────────────────────────────────────────
function createTarget() {
  const angle = rand(0, Math.PI * 2);
  const r     = rand(0.1, 0.82);
  const spd   = rand(0.002, 0.013);
  return {
    id:    STATE.nextId++,
    x:     0.5 + Math.cos(angle) * r * 0.45,
    y:     0.5 + Math.sin(angle) * r * 0.45,
    vx:    Math.cos(angle + Math.PI / 2) * spd * rand(0.5, 1.5),
    vy:    Math.sin(angle + Math.PI / 2) * spd * rand(0.5, 1.5),
    type:  Math.floor(rand(0, 4)),
    rcs:   rand(-10, 20),
    age:   0,
    trail: [],
  };
}

function initScene() {
  STATE.targets = [];
  for (let i = 0; i < STATE.numTargets; i++) STATE.targets.push(createTarget());

  STATE.clutter = [];
  for (let i = 0; i < 35; i++) {
    STATE.clutter.push({ x: rand(0.05, 0.95), y: rand(0.05, 0.95) });
  }

  STATE.sonarTargets = [];
  for (let i = 0; i < 4; i++) {
    STATE.sonarTargets.push({
      beam:   rand(10, 118),
      range:  rand(0.2, 0.9),
      vbeam:  rand(-0.25, 0.25),
      vrange: rand(-0.008, 0.008),
    });
  }

  STATE.trackHistory = {};
  STATE.frame       = 0;
  STATE.sweepAngle  = 0;
}

initScene();

// ── Canvas Setup ───────────────────────────────────────────────────────────────
const CANVASES = {};

function setupCanvas(id) {
  const el  = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const w   = el.parentElement.clientWidth || 600;
  const h   = parseInt(el.getAttribute('height'), 10);
  el.width  = Math.round(w * dpr);
  el.height = Math.round(h * dpr);
  el.style.height = h + 'px';
  const ctx = el.getContext('2d');
  ctx.scale(dpr, dpr);
  CANVASES[id] = { el, ctx, w, h };
}

function setupAll() {
  ['c-radar', 'c-sonar', 'c-rd', 'c-track'].forEach(setupCanvas);
}

setupAll();
window.addEventListener('resize', setupAll);

// ── Draw: Radar PPI ────────────────────────────────────────────────────────────
function drawRadar() {
  const { ctx, w, h } = CANVASES['c-radar'];
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = THEME.bg;
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2, cy = h / 2;
  const R  = Math.min(w, h) / 2 - 12;

  // Grid rings
  ctx.strokeStyle = THEME.grid;
  ctx.lineWidth = 0.5;
  for (let i = 1; i <= 4; i++) {
    ctx.beginPath();
    ctx.arc(cx, cy, R * i / 4, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
  ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
  ctx.stroke();

  // Sweep fill
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(STATE.sweepAngle);
  const arcLen = Math.PI * 0.55;
  const grd = ctx.createRadialGradient(0, 0, 2, 0, 0, R);
  grd.addColorStop(0, 'rgba(0,220,100,0.0)');
  grd.addColorStop(1, 'rgba(0,220,100,0.13)');
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, R, -arcLen, 0);
  ctx.closePath();
  ctx.fillStyle = grd;
  ctx.fill();

  // Sweep line
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(R, 0);
  ctx.strokeStyle = 'rgba(0,220,100,0.85)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();

  // Range labels
  ctx.fillStyle = THEME.text;
  ctx.font = '9px ' + (getComputedStyle(document.body).fontFamily || 'sans-serif');
  ctx.textAlign = 'center';
  for (let i = 1; i <= 4; i++) {
    ctx.fillText((i * 25) + 'm', cx + R * i / 4, cy - 4);
  }

  // Clutter
  if (STATE.noiseLevel > 10) {
    STATE.clutter.forEach(p => {
      const px = cx + (p.x - 0.5) * R * 2;
      const py = cy + (p.y - 0.5) * R * 2;
      if ((px - cx) ** 2 + (py - cy) ** 2 > R * R) return;
      ctx.beginPath();
      ctx.arc(px, py, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(136,135,128,${0.12 + STATE.noiseLevel / 350})`;
      ctx.fill();
    });
  }

  // Targets (radar + fused modes)
  if (STATE.mode !== 'sonar') {
    STATE.targets.forEach(t => {
      const px = cx + (t.x - 0.5) * R * 2;
      const py = cy + (t.y - 0.5) * R * 2;
      if ((px - cx) ** 2 + (py - cy) ** 2 > R * R) return;

      // Trail (phase 2+)
      if (STATE.phase >= 2 && t.trail.length > 1) {
        ctx.beginPath();
        ctx.moveTo(cx + (t.trail[0].x - 0.5) * R * 2, cy + (t.trail[0].y - 0.5) * R * 2);
        t.trail.forEach(pt => ctx.lineTo(cx + (pt.x - 0.5) * R * 2, cy + (pt.y - 0.5) * R * 2));
        ctx.strokeStyle = THEME.types[t.type] + '55';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Velocity arrow (phase 2+)
      if (STATE.phase >= 2) {
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + t.vx * R * 14, py + t.vy * R * 14);
        ctx.strokeStyle = THEME.types[t.type] + 'AA';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Dot
      const sz = t.type === 2 ? 5 : t.type === 0 ? 3 : 4;
      ctx.beginPath();
      ctx.arc(px, py, sz, 0, Math.PI * 2);
      ctx.fillStyle = THEME.types[t.type];
      ctx.fill();

      // Track label (phase 3)
      if (STATE.phase >= 3) {
        ctx.fillStyle = THEME.types[t.type];
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('T' + t.id, px + sz + 3, py + 3);
      }
    });
  }

  // Sonar overlay in fused mode
  if (STATE.mode === 'fused') {
    STATE.sonarTargets.forEach(st => {
      const angle = (st.beam / 128) * Math.PI * 2;
      const px = cx + Math.cos(angle) * st.range * R;
      const py = cy + Math.sin(angle) * st.range * R;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.strokeStyle = THEME.sonar;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }
}

// ── Draw: Sonar ────────────────────────────────────────────────────────────────
function drawSonar() {
  const { ctx, w, h } = CANVASES['c-sonar'];
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = THEME.bg;
  ctx.fillRect(0, 0, w, h);

  const beams = 128;
  const bw    = w / beams;

  // Noise background per beam
  for (let b = 0; b < beams; b++) {
    const noise = Math.random() * STATE.noiseLevel / 220;
    const x = b * bw;
    const grd = ctx.createLinearGradient(x, 0, x, h);
    grd.addColorStop(0, `rgba(30,60,80,${0.04 + noise})`);
    grd.addColorStop(1, `rgba(10,20,30,${0.02 + noise * 0.4})`);
    ctx.fillStyle = grd;
    ctx.fillRect(x, 0, bw, h);
  }

  // Sonar targets (sonar + fused modes)
  if (STATE.mode !== 'radar') {
    STATE.sonarTargets.forEach(st => {
      const bx = st.beam * bw + bw / 2;
      const ry = st.range * h;
      const intensity = 0.65 + 0.35 * Math.sin(STATE.frame * 0.05 + st.beam * 0.1);

      // Beam highlight
      const bGrd = ctx.createLinearGradient(bx - bw * 4, 0, bx + bw * 4, 0);
      bGrd.addColorStop(0, 'transparent');
      bGrd.addColorStop(0.5, `rgba(212,88,90,${0.1 * intensity})`);
      bGrd.addColorStop(1, 'transparent');
      ctx.fillStyle = bGrd;
      ctx.fillRect(bx - bw * 5, 0, bw * 10, h);

      // Echo blob
      const rGrd = ctx.createRadialGradient(bx, ry, 0, bx, ry, bw * 5);
      rGrd.addColorStop(0, `rgba(212,88,90,${0.9 * intensity})`);
      rGrd.addColorStop(0.5, `rgba(212,88,90,${0.25 * intensity})`);
      rGrd.addColorStop(1, 'transparent');
      ctx.fillStyle = rGrd;
      ctx.fillRect(bx - bw * 5, ry - 8, bw * 10, 16);

      ctx.beginPath();
      ctx.arc(bx, ry, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#ff8888';
      ctx.fill();

      ctx.fillStyle = THEME.sonar;
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('B' + Math.round(st.beam), bx, ry - 10);
    });
  }

  // Axes
  ctx.strokeStyle = THEME.grid;
  ctx.lineWidth = 0.5;
  for (let i = 1; i <= 4; i++) {
    const y = (i / 4) * h;
    ctx.beginPath();
    ctx.moveTo(0, y); ctx.lineTo(w, y);
    ctx.stroke();
    ctx.fillStyle = THEME.text;
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText((i * 25) + 'm', 4, y - 3);
  }
  ctx.fillStyle = THEME.text;
  ctx.textAlign = 'center';
  ctx.font = '10px sans-serif';
  ctx.fillText('← Beam index (0–128) →', w / 2, h - 4);
}

// ── Draw: Range-Doppler Map ─────────────────────────────────────────────────────
function drawRangeDoppler() {
  const { ctx, w, h } = CANVASES['c-rd'];
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = isDark ? '#0a1520' : '#e8f0f8';
  ctx.fillRect(0, 0, w, h);

  // Noise floor
  for (let x = 0; x < w; x += 2) {
    for (let y = 0; y < h; y += 2) {
      const n = Math.random() * STATE.noiseLevel / 200;
      if (n > 0.055) {
        ctx.fillStyle = `rgba(30,100,180,${n})`;
        ctx.fillRect(x, y, 2, 2);
      }
    }
  }

  // Target signatures
  if (STATE.mode !== 'sonar') {
    STATE.targets.forEach(t => {
      const range   = Math.sqrt((t.x - 0.5) ** 2 + (t.y - 0.5) ** 2);
      const doppler = t.vx * 0.5 + t.vy * 0.5;
      const rdx = clamp(range * w * 1.8, 5, w - 5);
      const rdy = clamp((doppler + 0.04) / 0.08 * h, 5, h - 5);

      const sig = ctx.createRadialGradient(rdx, rdy, 0, rdx, rdy, 20);
      sig.addColorStop(0,   THEME.types[t.type] + 'FF');
      sig.addColorStop(0.4, THEME.types[t.type] + '88');
      sig.addColorStop(1,   'transparent');
      ctx.fillStyle = sig;
      ctx.fillRect(rdx - 22, rdy - 22, 44, 44);

      ctx.beginPath();
      ctx.arc(rdx, rdy, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
    });
  }

  // Zero-Doppler line
  const zeroY = h / 2;
  ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)';
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = THEME.text;
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('v=0', 4, zeroY - 3);
  ctx.fillText('← Range →', 4, h - 4);

  ctx.save();
  ctx.translate(12, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('← Doppler →', 0, 0);
  ctx.restore();
}

// ── Draw: Tracking ─────────────────────────────────────────────────────────────
function drawTracking() {
  const { ctx, w, h } = CANVASES['c-track'];
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = THEME.bg;
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = THEME.grid;
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 5; i++) {
    ctx.beginPath();
    ctx.moveTo(w * i / 5, 0); ctx.lineTo(w * i / 5, h);
    ctx.moveTo(0, h * i / 5); ctx.lineTo(w, h * i / 5);
    ctx.stroke();
  }

  STATE.targets.forEach(t => {
    const px = t.x * w;
    const py = t.y * h;

    // Update track history
    if (!STATE.trackHistory[t.id]) STATE.trackHistory[t.id] = [];
    STATE.trackHistory[t.id].push({ x: px, y: py });
    if (STATE.trackHistory[t.id].length > 45) STATE.trackHistory[t.id].shift();

    // Trail
    const hist = STATE.trackHistory[t.id];
    if (hist.length > 1) {
      ctx.beginPath();
      ctx.moveTo(hist[0].x, hist[0].y);
      hist.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.strokeStyle = THEME.types[t.type] + '60';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Kalman prediction ellipse (phase 3)
    if (STATE.phase >= 3) {
      const predX = px + t.vx * w * 12;
      const predY = py + t.vy * h * 12;
      const angle = Math.atan2(t.vy, t.vx);
      ctx.beginPath();
      ctx.ellipse(predX, predY, 11, 7, angle, 0, Math.PI * 2);
      ctx.strokeStyle = THEME.types[t.type] + '55';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Bounding box
    const sz = t.type === 2 ? 14 : t.type === 0 ? 7 : 10;
    ctx.strokeStyle = THEME.types[t.type];
    ctx.lineWidth = 1.5;
    ctx.strokeRect(px - sz, py - sz, sz * 2, sz * 2);

    // Labels
    ctx.fillStyle = THEME.types[t.type];
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    const vel = (Math.sqrt(t.vx ** 2 + t.vy ** 2) * 100).toFixed(1);
    ctx.fillText('T' + t.id + ' ' + THEME.names[t.type], px + sz + 3, py + 3);
    ctx.fillText('v=' + vel + 'm/s', px + sz + 3, py + 14);
  });
}

// ── Update Simulation ───────────────────────────────────────────────────────────
function updateTargets() {
  // Sync target count
  while (STATE.targets.length < STATE.numTargets) STATE.targets.push(createTarget());
  while (STATE.targets.length > STATE.numTargets) STATE.targets.pop();

  STATE.targets.forEach(t => {
    t.trail.push({ x: t.x, y: t.y });
    if (t.trail.length > 28) t.trail.shift();

    t.x += t.vx * STATE.speed * 0.5;
    t.y += t.vy * STATE.speed * 0.5;
    t.age++;

    // Bounce
    if (t.x < 0.02 || t.x > 0.98) t.vx *= -1;
    if (t.y < 0.02 || t.y > 0.98) t.vy *= -1;
    t.x = clamp(t.x, 0.02, 0.98);
    t.y = clamp(t.y, 0.02, 0.98);
  });

  STATE.sonarTargets.forEach(st => {
    st.beam  = clamp(st.beam  + st.vbeam  * STATE.speed, 5, 123);
    st.range = clamp(st.range + st.vrange * STATE.speed, 0.1, 0.9);
    if (st.beam  < 5   || st.beam  > 123) st.vbeam  *= -1;
    if (st.range < 0.1 || st.range > 0.9) st.vrange *= -1;
  });
}

function updateMetrics() {
  const gpuUtil = Math.min(99, 62 + STATE.targets.length * 2 + STATE.noiseLevel * 0.25 + (STATE.phase - 1) * 8);
  const mota    = Math.min(99, 84 + STATE.phase * 3 - STATE.noiseLevel * 0.08);

  document.getElementById('m-det').innerHTML  = STATE.targets.length + ' <span class="metric-unit">/ frame</span>';
  document.getElementById('m-trk').innerHTML  = STATE.targets.length + ' <span class="metric-unit">tracks</span>';
  document.getElementById('m-gpu').innerHTML  = gpuUtil.toFixed(0) + ' <span class="metric-unit">%</span>';
  document.getElementById('radar-hz').textContent = (13 + STATE.speed) + ' Hz';
  document.getElementById('sonar-hz').textContent = (STATE.speed * 50) + ' kHz';
  document.getElementById('mota-badge').textContent = 'MOTA: ' + mota.toFixed(0) + '%';
}

// ── GPU Log ─────────────────────────────────────────────────────────────────────
const LOG_TEMPLATES = [
  ['ok',   'cuSignal FFT kernel launched — 128-pt DFT'],
  ['info', 'CFAR threshold — {n} detections above noise floor'],
  ['info', 'cuML DBSCAN clustering — {n} clusters found'],
  ['ok',   'Kalman predict step — dt={dt}ms'],
  ['warn', 'Association gap: 1 track unmatched this frame'],
  ['info', 'Ego-motion compensation applied from odometry'],
  ['ok',   'Beamforming: 128 beams × 500kHz — {t}µs GPU time'],
  ['info', 'Range-Doppler FFT: {n}×128 bins — {t}µs'],
  ['ok',   'Fused estimate updated — {n} objects in scene'],
  ['info', 'cuDF groupby — {n} frames aggregated'],
];

const logLines = [
  { type: 'ok',   msg: '[00.000] cuSignal init OK — CUDA 12.2 — GPU: RTX series' },
  { type: 'info', msg: '[00.001] cuDF allocator ready — pinned memory pool: 2 GB' },
  { type: 'info', msg: '[00.002] cuML DBSCAN loaded — n_neighbors=5' },
];

function addLog() {
  const interval = Math.max(1, Math.round(28 / STATE.speed));
  if (STATE.frame % interval !== 0) return;

  const tmpl = LOG_TEMPLATES[Math.floor(Math.random() * LOG_TEMPLATES.length)];
  const t    = (STATE.frame / 60).toFixed(3);
  const msg  = '[' + t + '] ' + tmpl[1]
    .replace('{n}', STATE.targets.length)
    .replace('{dt}', (1000 / 15).toFixed(1))
    .replace('{t}', rand(120, 800).toFixed(0));

  logLines.push({ type: tmpl[0], msg });
  if (logLines.length > 7) logLines.shift();

  const el = document.getElementById('gpu-log');
  el.innerHTML = logLines
    .map(l => `<div class="log-line log-${l.type}">${l.msg}</div>`)
    .join('');
}

// ── RAF Loop ─────────────────────────────────────────────────────────────────────
function tick(ts) {
  if (!STATE.running) { requestAnimationFrame(tick); return; }

  // FPS tracking
  STATE.fpsFrames++;
  if (ts - STATE.fpsLastTs > 600) {
    STATE.fps = (STATE.fpsFrames / ((ts - STATE.fpsLastTs) / 1000)) | 0;
    document.getElementById('fps-badge').textContent   = STATE.fps + ' FPS';
    document.getElementById('gpu-badge').textContent   = '● GPU: ' + (STATE.fps > 18 ? 'Active' : 'Busy');
    STATE.fpsFrames = 0;
    STATE.fpsLastTs = ts;
  }

  STATE.sweepAngle += 0.018 * STATE.speed;
  STATE.frame++;

  updateTargets();
  addLog();
  drawRadar();
  drawSonar();
  drawRangeDoppler();
  drawTracking();
  updateMetrics();

  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);

// ── UI Controls ───────────────────────────────────────────────────────────────
document.getElementById('btn-pause').addEventListener('click', function () {
  STATE.running = !STATE.running;
  this.textContent = STATE.running ? 'Pause' : 'Resume';
  this.classList.toggle('active', STATE.running);
});

document.getElementById('btn-reset').addEventListener('click', () => {
  initScene();
  logLines.length = 0;
  logLines.push({ type: 'ok', msg: '[00.000] System reset — reinitialising pipeline...' });
});

function sliderSetup(sliderId, valId, suffix, onChange) {
  const slider = document.getElementById(sliderId);
  const valEl  = document.getElementById(valId);
  slider.addEventListener('input', () => {
    valEl.textContent = slider.value + (suffix || '');
    onChange(parseFloat(slider.value));
  });
}

sliderSetup('sl-targets', 'sv-targets', '',  v => { STATE.numTargets  = Math.round(v); });
sliderSetup('sl-noise',   'sv-noise',   '',  v => { STATE.noiseLevel  = v; });
sliderSetup('sl-speed',   'sv-speed',   '×', v => { STATE.speed       = v; });

function setMode(m) {
  STATE.mode = m;
  ['radar', 'sonar', 'fused'].forEach(x => {
    document.getElementById('btn-' + x).classList.toggle('active', x === m);
  });
}

function setPhase(p) {
  STATE.phase = p;
  const labels = { 1: 'Phase 1 — Data Integration', 2: 'Phase 2 — RT Processing', 3: 'Phase 3 — Tracking' };
  document.getElementById('phase-badge').textContent = labels[p];
  [1, 2, 3].forEach(i => {
    document.getElementById('btn-p' + i).classList.toggle('active', i === p);
  });
}

document.getElementById('btn-radar').addEventListener('click', () => setMode('radar'));
document.getElementById('btn-sonar').addEventListener('click', () => setMode('sonar'));
document.getElementById('btn-fused').addEventListener('click', () => setMode('fused'));
document.getElementById('btn-p1').addEventListener('click', () => setPhase(1));
document.getElementById('btn-p2').addEventListener('click', () => setPhase(2));
document.getElementById('btn-p3').addEventListener('click', () => setPhase(3));

// ── Real Data API ─────────────────────────────────────────────────────────────
/**
 * Call this from your Python/JS bridge to inject a real RadarScenes frame.
 *
 * Expected format (matches RadarScenes HDF5 columns):
 *   frame = {
 *     detections: [
 *       { range: 12.4, azimuth: 0.35, rcs: 8.2, velocity: 3.1, label_id: 0 },
 *       ...
 *     ],
 *     timestamp: 1234567890.123,
 *     sequence_id: 42
 *   }
 *
 * label_id mapping (RadarScenes):
 *   0=PassengerCar, 1=LargeVehicle, 2=Truck, 3=Bus, 6=Motorcycle, 7=Pedestrian
 */
window.injectRealFrame = function (frame) {
  if (!frame || !Array.isArray(frame.detections)) return;

  STATE.useRealData = true;
  STATE.targets = frame.detections.map((det, i) => {
    // Convert polar (range, azimuth) → normalised [0,1] cartesian
    const x = 0.5 + Math.cos(det.azimuth) * (det.range / 100) * 0.9;
    const y = 0.5 + Math.sin(det.azimuth) * (det.range / 100) * 0.9;

    // Map RadarScenes label_id → display type index
    const typeMap = { 0: 1, 1: 2, 2: 2, 3: 2, 6: 3, 7: 0 };
    const type = typeMap[det.label_id] !== undefined ? typeMap[det.label_id] : 1;

    return {
      id:    STATE.nextId++,
      x:     clamp(x, 0.02, 0.98),
      y:     clamp(y, 0.02, 0.98),
      vx:    (det.velocity * Math.cos(det.azimuth)) / 500,
      vy:    (det.velocity * Math.sin(det.azimuth)) / 500,
      type,
      rcs:   det.rcs,
      age:   0,
      trail: [],
    };
  });
};
