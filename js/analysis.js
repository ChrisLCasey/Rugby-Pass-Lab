// Pass analysis: turns raw phone sensor streams into pass metrics.
// Pure functions only (no DOM) so it can be unit-tested in Node.
//
// Input recording:
//   motion: [{ t (s), ag: [x,y,z]|null  (m/s², incl. gravity, device frame),
//                      al: [x,y,z]|null  (m/s², gravity removed, device frame),
//                      w:  [x,y,z]       (deg/s about device x, y, z) }]
//   orient: [{ t (s), q: [x,y,z,w] }]   device -> earth (z up) from DeviceOrientation
//   gSign:  +1 / -1 sign convention of the accelerometer (see estimateGravitySign)
import { G, D2R, R2D, add, sub, mul, dot, len, qRot, qConj, qSlerp } from './math.js';

const AXIS_VEC = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
const AXIS_IDX = { x: 0, y: 1, z: 2 };

export const DEFAULTS = {
  ballAxis: 'y',        // which phone axis is the ball's long axis (y = along the phone)
  speedScale: 1,        // phone-to-ball factor: a phone held in the hand is swung slower than a ball is thrown
  source: 'auto',       // 'auto' = the phone's own gravity-free acceleration when it has one, 'ag' = raw accelerometer + orientation
  releaseHeight: 1.3,   // metres above ground at release
  onsetAccel: 3.0,      // m/s² that counts as the start of the movement
  quietAccel: 1.8,      // m/s² below which the phone counts as still
  quietRate: 60,        // deg/s below which the phone counts as still
  quietTime: 0.25,      // s of stillness that marks the end of the pass
  quietLead: 0.15,      // s of stillness required before the pass starts
  clipAccel: 75,        // m/s² (~7.6 g): phones often saturate near here
  clipRate: 1900,       // deg/s: typical gyro range is ±2000
};

/** Orientation lookup with slerp. `orient` must be sorted by t. */
export function orientAt(orient, t, hint = { i: 0 }) {
  const n = orient.length;
  if (!n) return null;
  if (t <= orient[0].t) return orient[0].q;
  if (t >= orient[n - 1].t) return orient[n - 1].q;
  let i = Math.min(hint.i, n - 2);
  while (i > 0 && orient[i].t > t) i--;
  while (i < n - 2 && orient[i + 1].t < t) i++;
  hint.i = i;
  const a = orient[i], b = orient[i + 1];
  const span = b.t - a.t || 1e-6;
  return qSlerp(a.q, b.q, (t - a.t) / span);
}

/** Replace jittery event-arrival times with an even grid when sampling is steady. */
export function regularizeTimes(samples) {
  const n = samples.length;
  if (n < 12) return samples;
  const dts = [];
  for (let i = 1; i < n; i++) dts.push(samples[i].t - samples[i - 1].t);
  const sorted = [...dts].sort((a, b) => a - b);
  const med = sorted[n >> 1];
  if (!(med > 0) || sorted[n - 2] > 3 * med) return samples; // gaps: keep real timestamps
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const y = samples[i].t;
    sx += i; sy += y; sxx += i * i; sxy += i * y;
  }
  const b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const a = (sy - b * sx) / n;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (samples[i].t - (a + b * i)) ** 2;
  if (Math.sqrt(ss / n) > 0.5 * med) return samples;
  return samples.map((s, i) => ({ ...s, t: a + b * i }));
}

/**
 * Some browsers report the accelerometer with the opposite sign (iOS vs Android history).
 * With the phone still, accelerationIncludingGravity must equal R^T * (0,0,+g).
 * Returns +1, -1, or 1 when it can't tell.
 */
export function estimateGravitySign(stillMotion, orient) {
  // keep only readings that are genuinely still (|a| close to 1 g, hardly rotating): people shuffle during the countdown
  const withAg = stillMotion.filter((s) => s.ag && Math.abs(len(s.ag) - G) < 0.8 && len(s.w || [0, 0, 0]) < 30);
  if (withAg.length < 5 || !orient.length) return { sign: 1, confidence: 0 };
  const m = [0, 0, 0];
  for (const s of withAg) { m[0] += s.ag[0]; m[1] += s.ag[1]; m[2] += s.ag[2]; }
  const mean = mul(m, 1 / withAg.length);
  const tMid = withAg[withAg.length >> 1].t;
  const q = orientAt(orient, tMid);
  const expected = qRot(qConj(q), [0, 0, G]);
  const d = dot(mean, expected) / ((len(mean) * G) || 1);
  return { sign: d < 0 ? -1 : 1, confidence: Math.abs(d) };
}

/** Linear (gravity-free) acceleration in the device frame for one sample. */
export function linearDevice(s, q, gSign = 1, source = 'auto', alSign = gSign) {
  if (s.al && source !== 'ag') return mul(s.al, alSign);
  if (!s.ag) return s.al ? mul(s.al, alSign) : null;
  const gravDev = qRot(qConj(q), [0, 0, G]);
  return sub(mul(s.ag, gSign), gravDev);
}

/** Horizontal "straight ahead" reference from the stance pose (earth frame, unit [x,y]). */
export function forwardFromQuat(q, ballAxis = 'y') {
  const cand = [
    qRot(q, AXIS_VEC[ballAxis]),
    qRot(q, [0, 0, -1]),
    qRot(q, [1, 0, 0]),
  ];
  for (const c of cand) {
    const h = Math.hypot(c[0], c[1]);
    if (h > 0.45) return [c[0] / h, c[1] / h];
  }
  return [0, 1];
}

function indexAtTime(t, time, from = 0) {
  let i = from;
  while (i < t.length - 1 && t[i] < time) i++;
  return i;
}

function fail(reason, extra = {}) {
  return { ok: false, reason, ...extra };
}

export function analysePass(rec, options = {}) {
  const o = { ...DEFAULTS, ...options };
  const motion = regularizeTimes(rec.motion.filter((s) => s.ag || s.al));
  const orient = [...rec.orient].sort((a, b) => a.t - b.t);
  const gSign = rec.gSign ?? 1;
  if (motion.length < 25) return fail('tooShort');
  if (!orient.length) return fail('noOrientation');

  const n = motion.length;
  const t = motion.map((s) => s.t);
  const hint = { i: 0 };
  const qs = new Array(n);
  const aw = new Array(n);
  const mag = new Float64Array(n);
  const wmag = new Float64Array(n);
  // Phones report gravity-free acceleration two ways: the OS's own ("al") and raw accelerometer minus gravity
  // using the orientation ("ag"). Use one for the measurement and the other as a cross-check.
  const both = motion.filter((s) => s.al && s.ag).length > 0.9 * n;
  // Some browsers flip the sign of the phone's own acceleration differently from the raw accelerometer, and a
  // flipped sign turns an upward pass into a downward one. With both channels the right sign is whichever
  // makes them agree.
  let alSign = gSign;
  if (both) {
    let agree = 0;
    const h2 = { i: 0 };
    for (let i = 0; i < n; i++) {
      const q = orientAt(orient, t[i], h2);
      const viaAg = linearDevice(motion[i], q, gSign, 'ag');
      agree += motion[i].al[0] * viaAg[0] + motion[i].al[1] * viaAg[1] + motion[i].al[2] * viaAg[2];
    }
    alSign = agree < 0 ? -1 : 1;
  }
  const primary = o.source === 'ag' ? 'ag' : 'auto';
  const altSource = primary === 'ag' ? 'al' : 'ag';
  const awAlt = both ? new Array(n) : null;
  for (let i = 0; i < n; i++) {
    const q = orientAt(orient, t[i], hint);
    qs[i] = q;
    const ad = linearDevice(motion[i], q, gSign, primary, alSign);
    aw[i] = qRot(q, ad);
    mag[i] = len(ad);
    wmag[i] = len(motion[i].w || [0, 0, 0]);
    if (awAlt) awAlt[i] = qRot(q, linearDevice(motion[i], q, gSign, altSource, alSign));
  }

  // --- find the pass: the movement that contains the strongest push.
  // Anything before `notBefore` (the countdown, where people shuffle into their stance) cannot be the pass,
  // and the movement is bounded by stillness on both sides, so a stance adjustment is never mistaken for it.
  const first = rec.notBefore != null ? indexAtTime(t, rec.notBefore) : 0;
  let peakIdx = -1, peakMag = 0;
  for (let i = Math.max(first, 1); i < n - 1; i++) {
    const m3 = (mag[i - 1] + mag[i] + mag[i + 1]) / 3;
    if (m3 > peakMag) { peakMag = m3; peakIdx = i; }
  }
  if (peakIdx < 0 || peakMag < o.onsetAccel) return fail('noMotion');
  if (peakMag < 5) return fail('tooGentle', { peakG: peakMag / G });

  const isStill = (i) => mag[i] < o.quietAccel && wmag[i] < o.quietRate;

  // start: walk back from the peak to the last stretch of stillness
  let start = Math.max(first, 0);
  let hasQuietLead = false;
  {
    let runEnd = -1;
    for (let i = peakIdx; i >= 0; i--) {
      if (isStill(i)) {
        if (runEnd < 0) runEnd = i;
        if (t[runEnd] - t[i] >= o.quietLead) { start = runEnd; hasQuietLead = true; break; }
      } else runEnd = -1;
    }
  }

  // the still samples just before the movement give the sensor bias
  const leadIdx = [];
  if (hasQuietLead) {
    for (let i = indexAtTime(t, t[start] - 0.5); i <= start; i++) if (isStill(i)) leadIdx.push(i);
  }
  const hasLead = leadIdx.length >= 5;

  // end: walk forward from the peak to the first stretch of stillness
  let end = n - 1;
  let rested = false;
  {
    let runStart = -1;
    for (let i = peakIdx; i < n; i++) {
      if (isStill(i)) {
        if (runStart < 0) runStart = i;
        if (t[i] - t[runStart] >= o.quietTime) { end = runStart; rested = true; break; }
      } else runStart = -1;
    }
  }
  if (end - start < 8) return fail('tooShort');

  // --- integrate acceleration -> velocity (trapezoid), bias removed
  const m = end - start + 1;
  const integrate = (acc) => {
    let bias = [0, 0, 0];
    if (hasLead) {
      for (const i of leadIdx) bias = add(bias, acc[i]);
      bias = mul(bias, 1 / leadIdx.length);
    }
    const out = new Array(m);
    out[0] = [0, 0, 0];
    for (let k = 1; k < m; k++) {
      const i = start + k;
      const dt = t[i] - t[i - 1];
      out[k] = add(out[k - 1], mul(add(sub(acc[i - 1], bias), sub(acc[i], bias)), dt / 2));
    }
    // Rest-to-rest: the phone is still at both ends, so any leftover velocity is sensor drift.
    if (rested) {
      const tot = t[end] - t[start] || 1;
      const vEnd = out[m - 1];
      for (let k = 0; k < m; k++) out[k] = sub(out[k], mul(vEnd, (t[start + k] - t[start]) / tot));
    }
    return out;
  };
  const v = integrate(aw);
  let speedAlt = null;
  if (awAlt) speedAlt = Math.max(...integrate(awAlt).map((x) => len(x)));
  const speed = v.map((x) => len(x));
  let rk = 0;
  for (let k = 1; k < m; k++) if (speed[k] > speed[rk]) rk = k;
  const rel = start + rk;                     // index of release in the full arrays
  const scale = o.speedScale > 0 ? o.speedScale : 1;
  const vMeasured = v[rk];                    // what the phone itself did
  const vRel = mul(vMeasured, scale);         // estimated ball release velocity
  const releaseSpeed = speed[rk] * scale;
  if (speedAlt != null) speedAlt *= scale;

  // positions (for the hand path during the throw)
  const pos = new Array(m);
  pos[0] = [0, 0, 0];
  for (let k = 1; k < m; k++) {
    const dt = t[start + k] - t[start + k - 1];
    pos[k] = add(pos[k - 1], mul(add(v[k - 1], v[k]), dt / 2));
  }

  // --- launch geometry
  const hSpeed = Math.hypot(vRel[0], vRel[1]);
  const elevation = Math.atan2(vRel[2], hSpeed) * R2D;
  const fwd = forwardFromQuat(qs[start], o.ballAxis);
  const cr = fwd[0] * vRel[1] - fwd[1] * vRel[0];     // + = left of forward
  const dr = fwd[0] * vRel[0] + fwd[1] * vRel[1];
  const direction = -Math.atan2(cr, dr) * R2D;         // + = right of forward
  const heading = Math.atan2(vRel[1], vRel[0]);

  // --- spin about the ball's long axis
  const ai = AXIS_IDX[o.ballAxis];
  const winFrom = indexAtTime(t, t[rel] - 0.08, start);
  let sumAbs = 0, sumSigned = 0, sumTot = 0, cnt = 0;
  for (let i = winFrom; i <= rel; i++) {
    const w = motion[i].w || [0, 0, 0];
    sumAbs += Math.abs(w[ai]); sumSigned += w[ai]; sumTot += len(w); cnt++;
  }
  const axisRate = cnt ? sumAbs / cnt : 0;           // deg/s
  const signedRate = cnt ? sumSigned / cnt : 0;
  const totRate = cnt ? sumTot / cnt : 0;
  let peakRate = 0, peakAxisRate = 0;
  for (let i = start; i <= Math.min(end, rel + 3); i++) {
    const w = motion[i].w || [0, 0, 0];
    peakRate = Math.max(peakRate, len(w));
    peakAxisRate = Math.max(peakAxisRate, Math.abs(w[ai]));
  }
  const spinRpm = axisRate / 6;
  const spiral = totRate > 5 ? Math.min(1, axisRate / totRate) : 0;

  // --- force & timing
  let peakAccel = 0, clipped = false;
  for (let i = start; i <= end; i++) {
    peakAccel = Math.max(peakAccel, mag[i]);
    if (mag[i] > o.clipAccel) clipped = true;
  }
  const gyroClipped = peakAxisRate > o.clipRate || peakRate > o.clipRate;
  let pushStart = start;
  for (let i = start; i <= rel; i++) if (mag[i] > o.onsetAccel) { pushStart = i; break; }
  const pushMs = (t[rel] - t[pushStart]) * 1000;

  // --- ballistic flight from release (no air drag)
  const h = o.releaseHeight;
  const vz = vRel[2];
  const tFlight = (vz + Math.sqrt(vz * vz + 2 * G * h)) / G;
  const distance = hSpeed * tFlight;
  const apex = vz > 0 ? h + (vz * vz) / (2 * G) : h;

  // --- replay data (hand path up to release, then the flight is analytic)
  const samples = [];
  const stride = Math.max(1, Math.floor(rk / 70));
  for (let k = 0; k <= rk; k += stride) {
    const p = sub(pos[k], pos[rk]);
    samples.push({ t: t[start + k] - t[rel], p: [p[0], p[1], p[2] + h], q: qs[start + k] });
  }
  if (samples[samples.length - 1].t !== 0) samples.push({ t: 0, p: [0, 0, h], q: qs[rel] });

  const rateHz = (n - 1) / ((t[n - 1] - t[0]) || 1);
  const warnings = [];
  if (!rested) warnings.push('notRested');
  if (!hasLead) warnings.push('noLead');
  if (clipped) warnings.push('accelClipped');
  if (gyroClipped) warnings.push('gyroClipped');
  if (rateHz < 30) warnings.push('lowRate');
  if (releaseSpeed > 38) warnings.push('implausibleSpeed');
  if (releaseSpeed < 2.5) warnings.push('veryGentle');
  if (speedAlt != null && Math.abs(speedAlt - releaseSpeed) / Math.max(releaseSpeed, 1) > 0.2) warnings.push('channelsDisagree');

  return {
    ok: true,
    speed: releaseSpeed,                // m/s
    speedMeasured: speed[rk],           // m/s the phone itself reached, before the phone-to-ball factor
    speedScale: scale,
    vMeasured,
    speedAlt,                           // m/s from the other acceleration channel (null if the phone only has one)
    source: primary === 'ag' ? 'ag' : (motion[0].al ? 'al' : 'ag'),
    elevation,                          // deg above horizontal
    direction,                          // deg, + = right of the stance direction
    spinRpm,
    peakSpinRpm: peakAxisRate / 6,
    spinSigned: signedRate,             // deg/s about the ball axis (sign = right-hand rule on the phone axis)
    spiral,                             // 0..1 share of rotation that is about the long axis
    peakG: peakAccel / G,
    pushMs,
    hangTime: tFlight,
    distance,
    apex,
    releaseHeight: h,
    vRelease: vRel,
    heading,
    forward: fwd,
    ballAxis: o.ballAxis,
    rateHz,
    rested,
    warnings,
    replay: {
      samples,
      flight: { p0: [0, 0, h], v: vRel, T: tFlight, g: G },
      spinRate: signedRate,
      handDuration: -samples[0].t,
    },
  };
}

export function classifyPass(r) {
  if (r.spinRpm >= 180 && r.spiral >= 0.6) return 'Spiral pass';
  if (r.spinRpm >= 60 && r.spiral >= 0.4) return 'Light spin pass';
  if (r.spinRpm < 60) return 'Flat pass';
  return 'Wobbly pass';
}

export const WARNING_TEXT = {
  notRested: "The phone didn't come to rest after the pass, so speed is less certain. Finish with a firm stop.",
  noLead: 'The phone was not still before the pass, so sensor drift could not be removed.',
  accelClipped: 'Peak force hit the sensor limit, so speed may read low.',
  gyroClipped: 'Spin hit the gyroscope limit (about 330 rpm on most phones), so it may read low.',
  lowRate: 'This browser delivered sensor readings slowly. Results are rougher than normal.',
  implausibleSpeed: 'That speed looks too high. Check the phone was held still before the pass.',
  veryGentle: 'Very gentle pass. Try a firmer movement for a more reliable reading.',
  channelsDisagree: "The phone's two acceleration readings disagree by more than 20%, so the speed is uncertain. Try the other acceleration source in Settings and compare.",
};

export const FAIL_TEXT = {
  tooShort: 'Not enough movement data was recorded. Try again.',
  noOrientation: "This device isn't reporting orientation. Check motion access is allowed for this site.",
  noMotion: 'No pass movement detected. Try a firmer pass, or raise the sensitivity in Settings.',
  tooGentle: 'That pass was too gentle to measure. Pass with more force.',
};

/** Flight time and range for a release velocity [vx, vy, vz] (earth frame, z up) from height h, ignoring drag. */
export function flightFor(v, h) {
  const hs = Math.hypot(v[0], v[1]);
  const T = (v[2] + Math.sqrt(v[2] * v[2] + 2 * G * h)) / G;
  return { T, distance: hs * T };
}

/** Phone-to-ball factor that makes a measured phone velocity travel `target` metres. Null if it can't. */
export function solveSpeedScale(vMeasured, h, target) {
  if (!(target > 0) || Math.hypot(vMeasured[0], vMeasured[1]) < 0.05) return null;
  let lo = 0.1, hi = 30;
  if (flightFor(mul(vMeasured, hi), h).distance < target) return null;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    if (flightFor(mul(vMeasured, mid), h).distance < target) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
