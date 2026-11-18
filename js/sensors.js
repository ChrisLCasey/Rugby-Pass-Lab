// Browser motion sensors: permission, capture buffers, live orientation.
import { qFromDeviceEuler } from './math.js';

const now = () => performance.now() / 1000;

export class Sensors {
  constructor() {
    this.motion = [];
    this.orient = [];
    this.latestQ = null;
    this.lastOrientAt = 0;
    this.lastMotionAt = 0;
    this.motionCount = 0;
    this.running = false;
    this.onMotion = null;     // (sample) => void, called for every motion reading
    this._m = (e) => this._handleMotion(e);
    this._o = (e) => this._handleOrientation(e);
  }

  get supported() {
    return typeof window !== 'undefined' && 'DeviceMotionEvent' in window;
  }

  /** iOS 13+ asks the user; other browsers just work (over HTTPS). */
  get needsPermission() {
    return typeof DeviceMotionEvent !== 'undefined' &&
      typeof DeviceMotionEvent.requestPermission === 'function';
  }

  /** Call directly from a tap handler (iOS only accepts it from a user gesture). */
  async requestPermission() {
    if (!this.supported) return 'unsupported';
    if (!this.needsPermission) return 'granted';
    // Newer Chrome also exposes requestPermission() and may answer "prompt", so only an explicit
    // "denied" (or a thrown error) counts as a refusal. Whether readings really arrive is checked by the caller.
    const ask = (p) => Promise.race([p, new Promise((r) => setTimeout(() => r('timeout'), 20000))]);
    try {
      const m = await ask(DeviceMotionEvent.requestPermission());
      let o = 'granted';
      if (typeof DeviceOrientationEvent !== 'undefined' &&
          typeof DeviceOrientationEvent.requestPermission === 'function') {
        o = await ask(DeviceOrientationEvent.requestPermission());
      }
      return m === 'denied' || o === 'denied' ? 'denied' : 'granted';
    } catch (err) {
      return 'denied';
    }
  }

  start() {
    if (this.running || !this.supported) return;
    window.addEventListener('devicemotion', this._m);
    window.addEventListener('deviceorientation', this._o);
    this.running = true;
  }

  stop() {
    window.removeEventListener('devicemotion', this._m);
    window.removeEventListener('deviceorientation', this._o);
    this.running = false;
  }

  clear() {
    this.motion = [];
    this.orient = [];
  }

  /** Keep only the last `seconds` of data (used while idle so memory stays flat). */
  trim(seconds) {
    const cut = now() - seconds;
    while (this.motion.length && this.motion[0].t < cut) this.motion.shift();
    while (this.orient.length && this.orient[0].t < cut) this.orient.shift();
  }

  /** True when readings arrived in the last `within` seconds. */
  isLive(within = 0.6) {
    return this.lastOrientAt > 0 && now() - this.lastOrientAt < within;
  }

  _handleMotion(e) {
    const t = now();
    const al = e.acceleration;
    const ag = e.accelerationIncludingGravity;
    const r = e.rotationRate;
    const hasAl = al && al.x != null && al.y != null && al.z != null;
    const hasAg = ag && ag.x != null && ag.y != null && ag.z != null;
    if (!hasAl && !hasAg) return;
    const s = {
      t,
      al: hasAl ? [al.x, al.y, al.z] : null,
      ag: hasAg ? [ag.x, ag.y, ag.z] : null,
      // W3C: alpha = about z, beta = about x, gamma = about y  ->  store as [x, y, z]
      w: r && r.alpha != null ? [r.beta || 0, r.gamma || 0, r.alpha || 0] : [0, 0, 0],
    };
    this.lastMotionAt = t;
    this.motionCount++;
    this.motion.push(s);
    if (this.onMotion) this.onMotion(s);
  }

  _handleOrientation(e) {
    if (e.alpha == null || e.beta == null || e.gamma == null) return;
    const t = now();
    const q = qFromDeviceEuler(e.alpha, e.beta, e.gamma);
    this.latestQ = q;
    this.lastOrientAt = t;
    this.orient.push({ t, q });
  }
}

/** Gravity-free magnitude of a sample, independent of any axis or sign convention. */
export function linMagnitude(s) {
  if (s.al) return Math.hypot(s.al[0], s.al[1], s.al[2]);
  return Math.abs(Math.hypot(s.ag[0], s.ag[1], s.ag[2]) - 9.80665);
}

export function rateMagnitude(s) {
  return Math.hypot(s.w[0], s.w[1], s.w[2]);
}
