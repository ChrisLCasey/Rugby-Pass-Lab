import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analysePass, estimateGravitySign, regularizeTimes, classifyPass } from '../js/analysis.js';
import { generateDemoPass } from '../js/demo.js';
import { G } from '../js/math.js';

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: got ${a.toFixed(3)}, want ${b} ±${tol}`);

test('recovers speed, launch angle, direction and spin of a clean pass', () => {
  const rec = generateDemoPass({ speed: 12, elevation: 28, direction: 6, spinRpm: 240 });
  const r = analysePass(rec);
  assert.ok(r.ok, r.reason);
  near(r.speed, 12, 0.6, 'speed');
  near(r.elevation, 28, 1.5, 'elevation');
  near(r.direction, 6, 2, 'direction');
  near(r.spinRpm, 240, 25, 'spin');
  assert.ok(r.spiral > 0.9, 'spiral share should be high for pure axial spin');
  assert.ok(r.rested);
  assert.equal(classifyPass(r), 'Spiral pass');
});

test('negative direction (pass to the left) and flat pass', () => {
  const rec = generateDemoPass({ speed: 15, elevation: 5, direction: -14, spinRpm: 20, seed: 3 });
  const r = analysePass(rec);
  assert.ok(r.ok, r.reason);
  near(r.direction, -14, 2, 'direction');
  near(r.elevation, 5, 1.5, 'elevation');
  assert.equal(classifyPass(r), 'Flat pass');
});

test('works with gravity-included acceleration only (no linear acceleration channel)', () => {
  const rec = generateDemoPass({ speed: 10, elevation: 20, withAl: false, seed: 11 });
  const r = analysePass(rec);
  assert.ok(r.ok, r.reason);
  near(r.speed, 10, 0.8, 'speed');
  near(r.elevation, 20, 2, 'elevation');
});

test('detects inverted accelerometer sign from a still window and corrects it', () => {
  const rec = generateDemoPass({ speed: 12, elevation: 25, deviceSign: -1, seed: 5 });
  const still = rec.motion.filter((s) => s.t > rec.stillWindow[0] && s.t < rec.stillWindow[1]);
  const est = estimateGravitySign(still, rec.orient);
  assert.equal(est.sign, -1);
  assert.ok(est.confidence > 0.9);
  const r = analysePass({ ...rec, gSign: est.sign });
  assert.ok(r.ok, r.reason);
  near(r.speed, 12, 0.7, 'speed');
  near(r.elevation, 25, 2, 'elevation');

  const normal = generateDemoPass({ seed: 5 });
  const est2 = estimateGravitySign(normal.motion.filter((s) => s.t > 0.1 && s.t < 0.8), normal.orient);
  assert.equal(est2.sign, 1);
});

test('rest-to-rest drift correction keeps speed accurate with sensor bias', () => {
  const rec = generateDemoPass({ speed: 12, elevation: 25, accelBias: 0.25, seed: 9 });
  const r = analysePass(rec);
  assert.ok(r.ok, r.reason);
  near(r.speed, 12, 1.0, 'speed with 0.25 m/s² bias');
});

test('ballistic maths are self-consistent', () => {
  const rec = generateDemoPass({ speed: 14, elevation: 30, direction: 0, seed: 21 });
  const r = analysePass(rec, { releaseHeight: 1.3 });
  assert.ok(r.ok, r.reason);
  const vz = r.vRelease[2], vh = Math.hypot(r.vRelease[0], r.vRelease[1]);
  const T = (vz + Math.sqrt(vz * vz + 2 * G * 1.3)) / G;
  near(r.hangTime, T, 1e-9, 'hang time');
  near(r.distance, vh * T, 1e-9, 'distance');
  near(r.apex, 1.3 + (vz * vz) / (2 * G), 1e-9, 'apex');
  assert.ok(r.replay.samples.length > 5);
  assert.deepEqual(r.replay.samples.at(-1).p, [0, 0, 1.3]);
});

test('ball axis option changes which gyro axis counts as spin', () => {
  const rec = generateDemoPass({ spinRpm: 240, seed: 2 });
  const r = analysePass(rec, { ballAxis: 'x' });
  assert.ok(!r.ok || r.spinRpm < 30, 'spin about phone-x should be tiny when the real spin is about y');
});

test('rejects recordings with no pass in them', () => {
  const rec = generateDemoPass({ speed: 12, seed: 4 });
  const quiet = rec.motion.filter((s) => s.t < 0.8);
  const r = analysePass({ ...rec, motion: quiet });
  assert.equal(r.ok, false);
  assert.equal(r.reason, 'noMotion');
  const r2 = analysePass({ ...rec, motion: rec.motion.slice(0, 10) });
  assert.equal(r2.reason, 'tooShort');
  const r3 = analysePass({ ...rec, orient: [] });
  assert.equal(r3.reason, 'noOrientation');
});

test('gentle passes are measured but flagged; near-still movements are rejected', () => {
  const r = analysePass(generateDemoPass({ speed: 1.5, seed: 8 }));
  assert.ok(r.ok, r.reason);
  near(r.speed, 1.5, 0.3, 'speed');
  assert.ok(r.warnings.includes('veryGentle'));
  const r2 = analysePass(generateDemoPass({ speed: 0.4, seed: 8 }));
  assert.equal(r2.ok, false);
  assert.ok(['tooGentle', 'noMotion'].includes(r2.reason));
});

test('timestamp regularisation removes arrival jitter but keeps real gaps', () => {
  const base = Array.from({ length: 100 }, (_, i) => ({ t: i * 0.01 + (i % 3) * 0.002 }));
  const reg = regularizeTimes(base);
  const dts = reg.slice(1).map((s, i) => s.t - reg[i].t);
  assert.ok(Math.max(...dts) - Math.min(...dts) < 1e-9);
  const gappy = base.map((s, i) => ({ t: i < 50 ? s.t : s.t + 0.2 }));
  assert.equal(regularizeTimes(gappy)[60].t, gappy[60].t);
});

test('survives heavy noise (3x) with a useful speed estimate', () => {
  const rec = generateDemoPass({ speed: 12, elevation: 25, noise: 3, seed: 13 });
  const r = analysePass(rec);
  assert.ok(r.ok, r.reason);
  near(r.speed, 12, 1.5, 'speed under 3x noise');
});

test('realistic phone sampling rates (60 Hz and 30 Hz) stay within a useful tolerance', () => {
  for (const [hz, tolSpeed, tolSpin] of [[60, 0.8, 30], [30, 1.5, 60]]) {
    const rec = generateDemoPass({ speed: 12, elevation: 28, direction: 6, spinRpm: 240, motionHz: hz, orientHz: hz, seed: 31 });
    const r = analysePass(rec);
    assert.ok(r.ok, r.reason);
    near(r.speed, 12, tolSpeed, `speed @${hz} Hz`);
    near(r.elevation, 28, 3, `elevation @${hz} Hz`);
    near(r.direction, 6, 3, `direction @${hz} Hz`);
    near(r.spinRpm, 240, tolSpin, `spin @${hz} Hz`);
    assert.ok(!r.warnings.includes('noLead'));
  }
});
