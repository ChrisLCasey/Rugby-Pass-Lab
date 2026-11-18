// Synthetic pass generator. Used by the "Demo pass" button (so the app can be tried on a laptop)
// and by the unit tests to check the analysis recovers known speed / angle / spin.
import { G, D2R, cross, norm, qFromBasis, qRot, qConj } from './math.js';

function gauss(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * @param {object} p
 *  speed (m/s), elevation (deg), direction (deg, + right of stance), spinRpm,
 *  motionHz, orientHz, noise (0..1 multiplier), accelBias (m/s², world z), seed,
 *  withAl (provide gravity-free acceleration), deviceSign (+1 or -1 simulates inverted accelerometers)
 */
export function generateDemoPass(p = {}) {
  const o = {
    speed: 12, elevation: 28, direction: 6, spinRpm: 240,
    motionHz: 100, orientHz: 60, noise: 1, accelBias: 0, seed: 7,
    withAl: true, deviceSign: 1, stanceElevation: 10,
    alDeviceSign: null,   // give the phone's own channel a different sign from the raw accelerometer
    alTau: 0,             // >0 simulates an OS gravity filter (seconds) that attenuates sustained pushes in `al`
    fidget: false,        // adjust the stance during the countdown before the pass, like a real user does
    ...p,
  };
  const rand = mulberry32(o.seed);
  const el = o.elevation * D2R, dr = o.direction * D2R;
  const dvec = [Math.sin(dr) * Math.cos(el), Math.cos(dr) * Math.cos(el), Math.sin(el)];

  const lead = o.fidget ? 2.2 : 0.9, Tb = 0.3, Tacc = 0.45, Td = 0.35, tail = 0.9;
  const vb = 0.12 * o.speed;
  const t1 = lead, t2 = t1 + Tb, t3 = t2 + Tacc, t4 = t3 + Td, total = t4 + tail;

  const sAcc = (t) => {                 // scalar acceleration along dvec
    if (t < t1) return 0;
    if (t < t2) return -vb * (Math.PI / Tb) * Math.cos((Math.PI * (t - t1)) / Tb);
    if (t < t3) return o.speed * (Math.PI / (2 * Tacc)) * Math.sin((Math.PI * (t - t2)) / Tacc);
    if (t < t4) return -o.speed * (Math.PI / (2 * Td)) * Math.sin((Math.PI * (t - t3)) / Td);
    return 0;
  };
  const omega = (t) => {                // deg/s about the phone's long (y) axis
    if (t < t3 - 0.2) return 0;
    if (t < t3) return o.spinRpm * 6 * Math.min(1, (t - (t3 - 0.2)) / 0.15);
    if (t < t3 + 0.1) return o.spinRpm * 6 * (1 - (t - t3) / 0.1);
    return 0;
  };

  // roll angle by fine numeric integration
  const fine = 1000, roll = new Float64Array(Math.ceil(total * fine) + 2);
  for (let i = 1; i < roll.length; i++) roll[i] = roll[i - 1] + omega(i / fine) * D2R / fine;
  const rollAt = (t) => roll[Math.min(roll.length - 1, Math.round(t * fine))];

  const stanceEl = o.stanceElevation * D2R;
  const ey = [0, Math.cos(stanceEl), Math.sin(stanceEl)];
  const ex0 = norm(cross(ey, [0, 0, 1]));
  const ez0 = cross(ex0, ey);
  const quatAt = (t) => {
    const psi = rollAt(t);
    const c = Math.cos(psi), s = Math.sin(psi);
    const ex = [c * ex0[0] - s * ez0[0], c * ex0[1] - s * ez0[1], c * ex0[2] - s * ez0[2]];
    const ez = cross(ex, ey);
    return qFromBasis(ex, ey, ez);
  };

  const motion = [];
  const dtm = 1 / o.motionHz;
  let gEst = null;
  for (let t = 0; t <= total; t += dtm) {
    const q = quatAt(t);
    const a = sAcc(t);
    let fx = 0;
    if (o.fidget && t > 0.3 && t < 0.9) {      // ~2 m/s sideways shuffle into the stance, 10 m/s² peak
      fx = t < 0.6 ? 2 * (Math.PI / 0.6) * Math.sin((Math.PI * (t - 0.3)) / 0.3) : -2 * (Math.PI / 0.6) * Math.sin((Math.PI * (t - 0.6)) / 0.3);
    }
    const aWorld = [dvec[0] * a + fx, dvec[1] * a, dvec[2] * a + o.accelBias];
    const nz = () => gauss(rand) * 0.12 * o.noise;
    const aNoisy = [aWorld[0] + nz(), aWorld[1] + nz(), aWorld[2] + nz()];
    let alDev = qRot(qConj(q), aNoisy);
    const agDev = qRot(qConj(q), [aNoisy[0], aNoisy[1], aNoisy[2] + G]);
    if (o.alTau > 0) {            // OS estimates gravity with a low-pass filter and subtracts it
      if (!gEst) gEst = qRot(qConj(q), [0, 0, G]);
      const k = dtm / o.alTau;
      const gNow = qRot(qConj(q), [0, 0, G]);
      // the estimate tracks the *measured* vector, so a long push leaks into it
      gEst = [0, 1, 2].map((j) => gEst[j] + (agDev[j] - gEst[j]) * k + (gNow[j] - gEst[j]) * 0);
      alDev = [0, 1, 2].map((j) => agDev[j] - gEst[j]);
    }
    const wn = () => gauss(rand) * 1.5 * o.noise;
    const w = [wn(), omega(t) + wn(), wn()];
    motion.push({
      t,
      al: o.withAl ? alDev.map((x) => x * (o.alDeviceSign ?? o.deviceSign)) : null,
      ag: agDev.map((x) => x * o.deviceSign),
      w,
    });
  }
  const orient = [];
  for (let t = 0; t <= total + 0.05; t += 1 / o.orientHz) orient.push({ t, q: quatAt(t) });

  return {
    motion, orient,
    gSign: 1,
    truth: { speed: o.speed, elevation: o.elevation, direction: o.direction, spinRpm: o.spinRpm },
    stillWindow: [o.fidget ? 1.2 : 0.1, lead - 0.1],
    goAt: lead - 0.1,
  };
}

/** A slightly different pass each press, so the demo isn't identical every time. */
export function randomDemoPass() {
  const r = Math.random;
  return generateDemoPass({
    speed: 9 + r() * 9,
    elevation: 12 + r() * 30,
    direction: -12 + r() * 24,
    spinRpm: 120 + r() * 200,
    seed: Math.floor(r() * 1e6),
  });
}
