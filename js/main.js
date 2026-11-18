import { BallViewer } from './viewer.js';
import { Sensors, linMagnitude, rateMagnitude } from './sensors.js';
import {
  analysePass, classifyPass, estimateGravitySign, forwardFromQuat, orientAt, WARNING_TEXT, FAIL_TEXT,
} from './analysis.js';
import { randomDemoPass } from './demo.js';
import {
  D2R, Q_IDENTITY, qMul, qAxis, qSlerp, qFromBasis, cross, G,
} from './math.js';

const $ = (id) => document.getElementById(id);
const els = {
  stage: $('stage'), gl: $('gl'), fx: $('fx'), hud: document.querySelector('.hud'),
  hudMain: $('hudMain'), hudSub: $('hudSub'), centre: $('btnCentre'),
  start: $('start'), demo: $('btnDemo'), notice: $('notice'),
  results: $('results'), flight: $('flight'), flightCap: $('flightCap'), warnings: $('warnings'),
  replay: $('btnReplay'), csv: $('btnCsv'),
  historyWrap: $('historyWrap'), history: $('history'), clear: $('btnClear'),
  dlg: $('dlgSettings'), settingsBtn: $('btnSettings'),
};

/* ------------------------------------------------------------------ settings */
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};
const settings = { units: 'kmh', axis: 'y', height: 1.3, trigger: 12, ...store.get('passlab.settings', {}) };
const UNITS = { kmh: { f: 3.6, l: 'km/h', d: 0 }, mph: { f: 2.23694, l: 'mph', d: 0 }, ms: { f: 1, l: 'm/s', d: 1 } };

function bindSettings() {
  $('setUnits').value = settings.units;
  $('setAxis').value = settings.axis;
  $('setHeight').value = settings.height;
  $('outHeight').textContent = Number(settings.height).toFixed(1);
  $('setTrigger').value = String(settings.trigger);
  const save = () => store.set('passlab.settings', settings);
  $('setUnits').onchange = (e) => { settings.units = e.target.value; save(); if (current) renderResult(current, false); renderHistory(); };
  $('setAxis').onchange = (e) => { settings.axis = e.target.value; save(); recentre(); };
  $('setHeight').oninput = (e) => { settings.height = Number(e.target.value); $('outHeight').textContent = settings.height.toFixed(1); save(); };
  $('setTrigger').onchange = (e) => { settings.trigger = Number(e.target.value); save(); };
  els.settingsBtn.onclick = () => els.dlg.showModal();
  els.dlg.addEventListener('close', () => { if (current && current.rec) reanalyse(); });
}

/* ------------------------------------------------------------------ scene pose helpers */
const Q_MAP = { y: Q_IDENTITY, x: qAxis([0, 0, 1], -Math.PI / 2), z: qAxis([1, 0, 0], Math.PI / 2) };

function earthToScene(fwd) {
  const f = [fwd[0], fwd[1], 0], up = [0, 0, 1], r = cross(f, up);
  return qFromBasis([f[0], up[0], r[0]], [f[1], up[1], r[1]], [f[2], up[2], r[2]]);
}
const toScenePos = (p, fwd) => [
  p[0] * fwd[0] + p[1] * fwd[1],
  p[2],
  p[0] * fwd[1] * 1 - p[1] * fwd[0],   // r = f x up = (fy, -fx, 0)
];
const ballQ = (S, qdev, axis) => qMul(qMul(S, qdev), Q_MAP[axis]);

/* ------------------------------------------------------------------ viewer + sensors */
const viewer = new BallViewer(els.gl, els.fx);
const sensors = new Sensors();
let ballReady = false;
let liveFwd = [0, 1];
let liveCentred = false;
let drag = { q: Q_IDENTITY, active: false, x: 0, y: 0 };

function recentre() {
  if (sensors.latestQ) { liveFwd = forwardFromQuat(sensors.latestQ, settings.axis); liveCentred = true; }
}

function setupDrag() {
  els.gl.addEventListener('pointerdown', (e) => {
    if (sensors.isLive() || replay) return;
    drag.active = true; drag.x = e.clientX; drag.y = e.clientY; els.gl.setPointerCapture(e.pointerId);
  });
  els.gl.addEventListener('pointermove', (e) => {
    if (!drag.active) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    drag.q = qMul(qMul(qAxis([0, 1, 0], dx * 0.012), qAxis([1, 0, 0], dy * 0.012)), drag.q);
  });
  const end = () => { drag.active = false; };
  els.gl.addEventListener('pointerup', end);
  els.gl.addEventListener('pointercancel', end);
}

/* ------------------------------------------------------------------ HUD / notices */
function hud(main, sub = '', go = false) {
  els.hudMain.textContent = main;
  els.hudSub.textContent = sub;
  els.hud.classList.toggle('go', go);
}
function notice(text, kind = 'error') {
  if (!text) { els.notice.hidden = true; return; }
  els.notice.textContent = text;
  els.notice.className = 'notice' + (kind === 'info' ? ' info' : '');
  els.notice.hidden = false;
}

/* ------------------------------------------------------------------ audio / haptics */
let actx = null;
function beep(freq, dur = 0.12, vol = 0.25) {
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === 'suspended') actx.resume();
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = 'sine'; o.frequency.value = freq;
    g.gain.setValueAtTime(vol, actx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime + dur);
    o.connect(g); g.connect(actx.destination);
    o.start(); o.stop(actx.currentTime + dur + 0.02);
  } catch { /* audio unavailable */ }
}
const buzz = (ms) => { try { navigator.vibrate && navigator.vibrate(ms); } catch { /* unsupported */ } };

/* ------------------------------------------------------------------ pass state machine */
// idle -> countdown -> ready -> capturing -> (result | idle)
let state = 'idle';
let armedAt = 0, goAt = 0, countdownStep = -1;
const trig = { count: 0, triggeredAt: 0, quietSince: 0 };
let current = null;      // { result, rec, id }
let replay = null;

const now = () => performance.now() / 1000;

function setButton() {
  const busy = state !== 'idle';
  els.start.textContent = busy ? 'Cancel' : 'Start pass';
  els.start.classList.toggle('live', busy);
  els.demo.disabled = busy;
}

async function onStart() {
  if (state !== 'idle') { cancel(); return; }
  notice('');
  // Must run inside the tap on iPhone, so request before anything async.
  const perm = await sensors.requestPermission();
  if (perm === 'unsupported') {
    notice('This browser has no motion sensors. Open the page on your phone, or try a demo pass.');
    return;
  }
  if (perm === 'denied') {
    notice('Motion access was blocked. On iPhone, open Settings, then Safari, then Motion & Orientation Access, and reload this page.');
    return;
  }
  sensors.start();
  beep(1, 0.01, 0.0001); // unlock audio inside the tap
  // wait briefly for the first readings so we can tell if the sensors are real
  const t0 = performance.now();
  while (!sensors.isLive(1) && performance.now() - t0 < 1500) await new Promise((r) => setTimeout(r, 50));
  if (!sensors.isLive(1)) {
    notice("No sensor readings are arriving. Open this page over HTTPS on a phone, allow motion access, or try a demo pass.");
    return;
  }
  stopReplay();
  hideResults(false);
  recentre();
  sensors.clear();
  armedAt = now();
  state = 'countdown'; countdownStep = -1;
  setButton();
}

function cancel(msg) {
  state = 'idle';
  setButton();
  idleHud();
  if (msg) notice(msg, 'info');
}

function tickCountdown() {
  const e = now() - armedAt;
  const step = Math.min(3, Math.floor(e));      // 0,1,2 then 3 = go
  if (step === countdownStep) return;
  countdownStep = step;
  if (step < 3) {
    const n = 3 - step;
    hud(String(n), 'Get into your passing stance and stay still.');
    beep(520, 0.1);
  } else {
    goAt = armedAt + 3;
    state = 'ready';
    trig.count = 0; trig.quietSince = 0;
    hud('Pass now', 'Pass the phone like a ball, then finish with a firm stop.', true);
    beep(880, 0.25, 0.35); buzz(60);
  }
}

sensors.onMotion = (s) => {
  if (state === 'ready') {
    if (linMagnitude(s) > settings.trigger) {
      trig.count++;
      if (trig.count >= 2) {
        state = 'capturing'; trig.triggeredAt = s.t; trig.quietSince = 0;
        hud('Measuring', 'Follow through and finish still.');
      }
    } else trig.count = 0;
  } else if (state === 'capturing') {
    const quiet = linMagnitude(s) < 1.8 && rateMagnitude(s) < 60;
    if (quiet) { if (!trig.quietSince) trig.quietSince = s.t; } else trig.quietSince = 0;
    const settled = trig.quietSince && s.t - trig.quietSince > 0.4 && s.t - trig.triggeredAt > 0.5;
    if (settled || s.t - trig.triggeredAt > 3.5) finishCapture();
  }
};

function finishCapture() {
  state = 'idle';
  setButton();
  buzz(40);
  const still = sensors.motion.filter((s) => s.t > goAt - 0.95 && s.t < goAt - 0.15);
  const orient = sensors.orient.filter((o) => o.t >= armedAt - 0.1);
  const est = estimateGravitySign(still, orient);
  const rec = {
    motion: sensors.motion.filter((s) => s.t >= armedAt),
    orient,
    gSign: est.confidence > 0.6 ? est.sign : 1,
  };
  runAnalysis(rec);
}

function analysisOptions() {
  return { ballAxis: settings.axis, releaseHeight: Number(settings.height) };
}

function runAnalysis(rec, label = '') {
  hud('Working it out');
  const result = analysePass(rec, analysisOptions());
  if (!result.ok) {
    idleHud();
    notice(FAIL_TEXT[result.reason] || 'That pass could not be analysed. Try again.');
    return;
  }
  result.demo = label === 'demo';
  const entry = { id: String(Date.now()), result, rec, ts: Date.now() };
  current = entry;
  pushHistory(entry);
  renderResult(entry, false);
  startReplay(entry, true);
}

function reanalyse() {
  if (!current || !current.rec) return;
  const r = analysePass(current.rec, analysisOptions());
  if (r.ok) { r.demo = current.result.demo; current.result = r; renderResult(current, false); }
}

/* ------------------------------------------------------------------ results UI */
const fmt = (n, d = 0) => Number(n).toFixed(d);

function tween(el, to, dec, suffix = '') {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce) { el.textContent = fmt(to, dec) + suffix; return; }
  const t0 = performance.now(), dur = 650;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(to * e, dec) + suffix;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderResult(entry, animate) {
  const r = entry.result;
  const u = UNITS[settings.units];
  els.results.hidden = false;
  $('rType').textContent = classifyPass(r);
  $('rSub').textContent = r.demo
    ? 'Demo pass from simulated sensor data.'
    : new Date(entry.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  $('rSpeedU').textContent = u.l;
  const dirAbs = Math.abs(r.direction);
  $('rDirLab').textContent = dirAbs < 1.5 ? 'Direction, straight on' : `Direction, degrees ${r.direction > 0 ? 'right' : 'left'}`;
  const setNum = (id, v, d) => (animate ? tween($(id), v, d) : ($(id).textContent = fmt(v, d)));
  setNum('rSpeed', r.speed * u.f, u.d);
  setNum('rAngle', r.elevation, 0);
  setNum('rSpin', r.spinRpm, 0);
  setNum('rDir', dirAbs < 1.5 ? 0 : dirAbs, 0);
  $('rSpiral').textContent = `${fmt(r.spiral * 100)}%`;
  $('rSpiralBar').style.width = `${fmt(r.spiral * 100)}%`;
  $('rG').textContent = `${fmt(r.peakG, 1)} g`;
  $('rPush').textContent = `${fmt(r.pushMs)} ms`;
  $('rHang').textContent = `${fmt(r.hangTime, 1)} s`;
  $('rDist').textContent = `${fmt(r.distance, r.distance < 10 ? 1 : 0)} m`;
  $('rApex').textContent = `${fmt(r.apex, 1)} m`;
  els.flightCap.textContent = `Estimated flight from a ${fmt(r.releaseHeight, 1)} m release, ignoring air drag.`;
  els.warnings.replaceChildren(...r.warnings.map((w) => {
    const li = document.createElement('li'); li.textContent = WARNING_TEXT[w] || w; return li;
  }));
  els.csv.disabled = !entry.rec;
  requestAnimationFrame(() => drawFlight(r));
  renderHistory();
}

function hideResults(clearCurrent = true) {
  els.results.hidden = true;
  if (clearCurrent) current = null;
}

/* ---- flight diagram: side view drawn like a pitch ---- */
function drawFlight(r) {
  const c = els.flight;
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const W = c.clientWidth, H = c.clientHeight;
  if (!W) return;
  c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const padL = 14, padR = 16, groundY = H - 26;
  const xmax = Math.max(r.distance * 1.22, 8);
  const ymax = Math.max(r.apex * 1.15, 3.2);
  const sc = Math.min((W - padL - padR) / xmax, (groundY - 18) / ymax);
  const X = (m) => padL + m * sc, Y = (m) => groundY - m * sc;

  // pitch markings every 5 m
  ctx.font = '600 13px "Barlow Condensed", "Arial Narrow", sans-serif';
  ctx.textAlign = 'center';
  const step = xmax > 30 ? 10 : 5;
  for (let m = 0; m <= xmax; m += step) {
    const major = m % (step * 2) === 0 || step === 10;
    ctx.strokeStyle = major ? 'rgba(244,247,250,0.5)' : 'rgba(244,247,250,0.22)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(m), 8); ctx.lineTo(X(m), groundY); ctx.stroke();
    ctx.fillStyle = 'rgba(244,247,250,0.85)';
    ctx.fillText(m === 0 ? '0' : `${m} m`, Math.max(X(m), padL + 6), H - 8);
  }
  ctx.fillStyle = 'rgba(244,247,250,0.28)';
  ctx.fillRect(0, groundY, W, 2);

  // trajectory
  const vh = Math.hypot(r.vRelease[0], r.vRelease[1]), vz = r.vRelease[2], T = r.hangTime;
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.lineCap = 'round';
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const t = (T * i) / 60;
    const x = vh * t, y = Math.max(0, r.releaseHeight + vz * t - 0.5 * G * t * t);
    i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y));
  }
  ctx.stroke();
  // every quarter second
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  for (let t = 0.25; t < T; t += 0.25) {
    const x = vh * t, y = r.releaseHeight + vz * t - 0.5 * G * t * t;
    ctx.beginPath(); ctx.arc(X(x), Y(y), 2.5, 0, Math.PI * 2); ctx.fill();
  }
  // release and landing
  ctx.fillStyle = '#c8102e';
  ctx.beginPath(); ctx.arc(X(0), Y(r.releaseHeight), 6, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
  const lx = X(r.distance);
  ctx.fillStyle = '#c8102e';
  ctx.beginPath(); ctx.arc(lx, groundY, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = '700 20px "Barlow Condensed", "Arial Narrow", sans-serif';
  ctx.textAlign = 'left';
  const lab = `${fmt(r.distance, r.distance < 10 ? 1 : 0)} m`;
  ctx.fillText(lab, lx + 11, groundY - 9);
}

/* ------------------------------------------------------------------ history */
let history = store.get('passlab.history', []);
function slim(entry) {
  const round = (a) => a.map((x) => Math.round(x * 1e4) / 1e4);
  const r = entry.result;
  return {
    id: entry.id, ts: entry.ts,
    result: {
      ...r, vRelease: round(r.vRelease), forward: round(r.forward),
      replay: {
        ...r.replay,
        samples: r.replay.samples.map((s) => ({ t: Math.round(s.t * 1e4) / 1e4, p: round(s.p), q: round(s.q) })),
        flight: { ...r.replay.flight, v: round(r.replay.flight.v) },
      },
    },
  };
}
const raw = new Map();   // id -> rec (this session only)
function pushHistory(entry) {
  raw.set(entry.id, entry.rec);
  history.unshift(slim(entry));
  history = history.slice(0, 12);
  store.set('passlab.history', history);
}
function renderHistory() {
  els.historyWrap.hidden = history.length === 0;
  const u = UNITS[settings.units];
  const fastest = history.reduce((b, h) => (h.result.speed > (b?.result.speed ?? 0) ? h : b), null);
  els.history.replaceChildren(...history.map((h) => {
    const r = h.result;
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'h-item';
    if (current && current.id === h.id) b.setAttribute('aria-current', 'true');
    b.innerHTML = `<span class="h-main">${fmt(r.speed * u.f, u.d)} ${u.l}${h === fastest && history.length > 1 ? '<span class="h-tag">Fastest</span>' : ''}</span>
      <span class="h-time">${new Date(h.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
      <span class="h-meta">${fmt(r.elevation)}° up, ${fmt(r.spinRpm)} rpm, ${classifyPass(r).toLowerCase()}</span>`;
    b.onclick = () => {
      current = { id: h.id, result: r, rec: raw.get(h.id), ts: h.ts };
      renderResult(current, false);
      startReplay(current);
      els.stage.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    li.append(b);
    return li;
  }));
}

/* ------------------------------------------------------------------ CSV export */
function downloadCsv() {
  if (!current || !current.rec) return;
  const { motion, orient } = current.rec;
  const hint = { i: 0 };
  const t0 = motion[0].t;
  const rows = ['t_s,ax,ay,az,gx,gy,gz,wx_dps,wy_dps,wz_dps,qx,qy,qz,qw'];
  for (const s of motion) {
    const q = orientAt([...orient].sort((a, b) => a.t - b.t), s.t, hint) || [0, 0, 0, 1];
    const g = s.ag || ['', '', ''];
    const a = s.al || ['', '', ''];
    rows.push([(s.t - t0).toFixed(4), ...a, ...g, ...s.w, ...q.map((x) => x.toFixed(5))].join(','));
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `pass-${new Date(current.ts).toISOString().replace(/[:.]/g, '-')}.csv`;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ------------------------------------------------------------------ replay */
const REPLAY_RATE = 0.5;
function startReplay(entry, scrollAfter = false) {
  const rep = entry.result.replay;
  replay = {
    scrollAfter,
    rep, axis: entry.result.ballAxis || settings.axis, fwd: entry.result.forward,
    t0: performance.now(), tau0: -rep.handDuration - 0.15,
    end: rep.flight.T + 1.1, trail: [], lastTrailAt: -1,
  };
  hud('Replay', `Half speed. Ball released at ${fmt(entry.result.speed * UNITS[settings.units].f, UNITS[settings.units].d)} ${UNITS[settings.units].l}.`);
}
function stopReplay() { replay = null; }

function finishReplay() {
  const scroll = replay && replay.scrollAfter;
  replay = null;
  idleHud();
  if (scroll && current && window.scrollY < 160) {
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    renderResult(current, true);
    els.results.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  }
}

function replayPose(rp) {
  const tau = rp.tau0 + ((performance.now() - rp.t0) / 1000) * REPLAY_RATE;
  if (tau > rp.end) return null;
  const S = earthToScene(rp.fwd);
  const { samples, flight, spinRate } = rp.rep;
  let pE, qdev, spin = 0;
  if (tau < 0) {
    const t = Math.max(tau, samples[0].t);
    let i = 0;
    while (i < samples.length - 2 && samples[i + 1].t < t) i++;
    const a = samples[i], b = samples[i + 1];
    const k = b.t === a.t ? 0 : Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t)));
    pE = [0, 1, 2].map((j) => a.p[j] + (b.p[j] - a.p[j]) * k);
    qdev = qSlerp(a.q, b.q, k);
  } else {
    const t = Math.min(tau, flight.T);
    pE = [flight.p0[0] + flight.v[0] * t, flight.p0[1] + flight.v[1] * t, flight.p0[2] + flight.v[2] * t - 0.5 * flight.g * t * t];
    qdev = samples[samples.length - 1].q;
    spin = spinRate * D2R * t;
  }
  const pos = toScenePos(pE, rp.fwd);
  pos[1] = Math.max(pos[1], 0.1);
  const q = qMul(ballQ(S, qdev, rp.axis), qAxis([0, 1, 0], spin));
  return { pos, q, tau };
}

/* ------------------------------------------------------------------ idle HUD */
function idleHud() {
  if (current) { hud('Pass logged', 'Start another pass, or replay this one.'); return; }
  if (!ballReady) { hud('Loading ball'); return; }
  if (sensors.isLive()) hud('Tilt your phone', 'The ball follows your hand. Tap Start pass when you are ready.');
  else if (sensors.needsPermission) hud('Ready when you are', 'Tap Start pass to switch on the motion sensors.');
  else hud('Drag to turn the ball', 'No motion sensors here. Open this on your phone, or try a demo pass.');
}

/* ------------------------------------------------------------------ main loop */
let lastTrim = 0;
function frame(ts) {
  requestAnimationFrame(frame);
  if (state === 'countdown') tickCountdown();
  if (state === 'ready' && now() - goAt > 12) { cancel('No pass detected. Pass more firmly, or raise the trigger sensitivity in Settings.'); }
  if ((state === 'idle') && ts - lastTrim > 1000) { sensors.trim(3); lastTrim = ts; }

  let frameState;
  if (replay) {
    const p = replayPose(replay);
    if (!p) finishReplay();
    else {
      const rp = replay;
      if (p.tau >= -0.2 && ts - rp.lastTrailAt > 16) { rp.trail.push(p.pos); rp.lastTrailAt = ts; if (rp.trail.length > 200) rp.trail.shift(); }
      const dist = viewer.fitDistance(0.3);
      frameState = { q: p.q, pos: p.pos, dist, lookY: -dist * 0.1, ground: true, trail: rp.trail };
    }
  }
  if (!frameState) {
    const dist = viewer.fitDistance(0.78);
    if (sensors.isLive()) {
      if (!liveCentred) recentre();
      const S = earthToScene(liveFwd);
      els.centre.hidden = state !== 'idle' || !!replay;
      frameState = { q: ballQ(S, sensors.latestQ, settings.axis), pos: [0, 0, 0], dist, lookY: 0 };
    } else {
      els.centre.hidden = true;
      const t = ts / 1000;
      const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
      const base = qMul(qAxis([0, 0, 1], -0.95), qAxis([0, 1, 0], reduce ? 0.5 : t * 0.55));
      frameState = { q: qMul(drag.q, base), pos: [0, 0, 0], dist, lookY: 0 };
    }
  }
  viewer.draw(frameState);
}

/* ------------------------------------------------------------------ wiring */
async function init() {
  bindSettings();
  setupDrag();
  renderHistory();
  setButton();
  els.start.onclick = onStart;
  els.centre.onclick = recentre;
  els.demo.onclick = () => {
    if (state !== 'idle') return;
    notice('');
    stopReplay();
    runAnalysis(randomDemoPass(), 'demo');
  };
  els.replay.onclick = () => { if (current) { startReplay(current); els.stage.scrollIntoView({ behavior: 'smooth', block: 'start' }); } };
  els.csv.onclick = downloadCsv;
  els.clear.onclick = () => { history = []; store.set('passlab.history', history); raw.clear(); renderHistory(); };
  window.addEventListener('resize', () => { if (current) drawFlight(current.result); });
  document.addEventListener('visibilitychange', () => { if (document.hidden && state !== 'idle') cancel('Paused because the page was hidden.'); });

  if (!viewer.supported) {
    hud('3D view unavailable', 'This browser cannot draw the ball, but measuring passes still works.');
  } else {
    try {
      await viewer.load('assets/ball.glb', 'assets/ball_normal_os.jpg');
      ballReady = true;
    } catch (err) {
      console.error(err);
      hud('Ball failed to load', 'Check assets/ball.glb is deployed. Measuring passes still works.');
    }
  }
  // Listening is harmless before permission: Android/desktop deliver straight away, iPhone after the tap.
  if (sensors.supported) sensors.start();
  idleHud();
  requestAnimationFrame(frame);
  // refresh the idle message once we know whether sensors are really delivering
  setTimeout(() => { if (state === 'idle' && !current) idleHud(); }, 1600);

  // test hook (also handy for debugging in the console)
  window.__passlab = { analysePass, runAnalysis, sensors, get state() { return state; }, viewer };
}

init();
