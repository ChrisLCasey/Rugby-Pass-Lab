// Dependency-free WebGL2 viewer for the England ball, plus a 2D layer for the pitch,
// distance markers and flight trail. Scene axes: +X = pass direction (screen right), +Y up, +Z toward camera.
import {
  mat4Perspective, mat4LookAt, mat4Mul, mat4FromQuatPos, mat3FromQuat, transformPoint,
} from './math.js';

const VERT = `#version 300 es
in vec3 aPos; in vec3 aNrm; in vec2 aUv;
uniform mat4 uModel; uniform mat4 uViewProj; uniform mat3 uRot;
out vec3 vNrm; out vec2 vUv; out vec3 vPos;
void main(){
  vec4 w = uModel * vec4(aPos,1.0);
  vPos = w.xyz; vNrm = uRot * aNrm; vUv = aUv;
  gl_Position = uViewProj * w;
}`;

const FRAG = `#version 300 es
precision highp float;
in vec3 vNrm; in vec2 vUv; in vec3 vPos;
uniform sampler2D uBase; uniform sampler2D uNrm;
uniform mat3 uRot; uniform vec3 uCam; uniform float uRough; uniform float uHasNrm;
out vec4 outColor;

vec3 sky(vec3 d){
  vec3 top = vec3(0.50,0.58,0.86), hor = vec3(0.86,0.90,0.98), gnd = vec3(0.10,0.26,0.17);
  float h = d.y;
  vec3 c = mix(hor, top, smoothstep(0.0, 0.8, h));
  return mix(gnd, c, smoothstep(-0.25, 0.1, h));
}
float spec(vec3 N, vec3 V, vec3 L, float shin){
  vec3 H = normalize(V + L);
  float nh = max(dot(N,H),0.0);
  return pow(nh, shin) * (shin + 8.0) / 25.13 * max(dot(N,L),0.0);
}
void main(){
  vec3 base = texture(uBase, vUv).rgb;
  vec3 Ng = normalize(vNrm);
  vec3 N = Ng;
  if (uHasNrm > 0.5) {
    vec3 nos = texture(uNrm, vUv).rgb * 2.0 - 1.0;
    vec3 Nm = normalize(uRot * nos);
    N = dot(Nm, Ng) > 0.25 ? Nm : normalize(mix(Nm, Ng, 0.8));
  }
  vec3 V = normalize(uCam - vPos);
  float ndv = max(dot(N,V), 0.0);

  vec3 L1 = normalize(vec3(-0.45, 0.75, 0.55));   // key
  vec3 L2 = normalize(vec3( 0.80, 0.10, 0.45));   // fill
  vec3 L3 = normalize(vec3(-0.15, 0.35,-1.00));   // rim
  float d1 = max(dot(N,L1),0.0), d2 = max(dot(N,L2),0.0), d3 = max(dot(N,L3),0.0);
  vec3 amb = mix(vec3(0.10,0.22,0.15), vec3(0.46,0.52,0.72), N.y*0.5+0.5) * 0.85;
  vec3 diff = amb + vec3(1.00,0.97,0.92)*2.1*d1 + vec3(0.40,0.52,0.85)*0.55*d2 + vec3(0.85,0.90,1.0)*0.9*d3;
  vec3 col = base * diff;

  float shin = max(2.0 / pow(max(uRough,0.08),4.0) - 2.0, 4.0);
  shin = min(shin, 90.0);
  float F = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
  float sp = spec(N,V,L1,shin) * 1.6 + spec(N,V,L2,shin) * 0.4;
  vec3 R = reflect(-V, N);
  vec3 env = sky(R) * (F * 0.9 + 0.03) * (1.0 - uRough * 0.55);
  col += vec3(1.0) * sp * (0.04 + 0.5 * F) + env;

  col = col / (1.0 + col * 0.12);          // soft shoulder
  col = pow(col, vec3(1.0/2.2));
  outColor = vec4(col, 1.0);
}`;

function parseGLB(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('Not a GLB file');
  let off = 12, json = null, bin = null;
  while (off < buf.byteLength) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const data = buf.slice(off + 8, off + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(data));
    else if (type === 0x004e4942) bin = data;
    off += 8 + len;
  }
  return { json, bin };
}

function accessor({ json, bin }, idx) {
  const a = json.accessors[idx];
  const bv = json.bufferViews[a.bufferView];
  const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[a.type];
  const Ctor = { 5126: Float32Array, 5123: Uint16Array, 5125: Uint32Array }[a.componentType];
  return { data: new Ctor(bin, (bv.byteOffset || 0) + (a.byteOffset || 0), a.count * comps), comps, count: a.count, type: a.componentType };
}

export const BALL_LENGTH = 0.29;   // metres
const FOV = 34 * Math.PI / 180;

export class BallViewer {
  constructor(glCanvas, fxCanvas) {
    this.glCanvas = glCanvas;
    this.fx = fxCanvas;
    this.ctx = fxCanvas.getContext('2d');
    this.gl = glCanvas.getContext('webgl2', { alpha: true, antialias: true, premultipliedAlpha: true });
    this.ready = false;
    this.dpr = 1; this.w = 1; this.h = 1;
    this.vp = null;
    if (!this.gl) return;
    this._initGL();
  }

  get supported() { return !!this.gl; }

  _initGL() {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.bindAttribLocation(p, 0, 'aPos'); gl.bindAttribLocation(p, 1, 'aNrm'); gl.bindAttribLocation(p, 2, 'aUv');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    this.prog = p;
    this.u = {};
    for (const n of ['uModel', 'uViewProj', 'uRot', 'uBase', 'uNrm', 'uCam', 'uRough', 'uHasNrm']) {
      this.u[n] = gl.getUniformLocation(p, n);
    }
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.clearColor(0, 0, 0, 0);
  }

  async load(glbUrl, normalUrl) {
    if (!this.gl) return;
    const gl = this.gl;
    const buf = await (await fetch(glbUrl)).arrayBuffer();
    const glb = parseGLB(buf);
    const prim = glb.json.meshes[0].primitives[0];
    const pos = accessor(glb, prim.attributes.POSITION);
    const nrm = accessor(glb, prim.attributes.NORMAL);
    const uv = accessor(glb, prim.attributes.TEXCOORD_0);
    const idx = accessor(glb, prim.indices);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const attr = (loc, a) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, a.data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, a.comps, gl.FLOAT, false, 0, 0);
    };
    attr(0, pos); attr(1, nrm); attr(2, uv);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx.data, gl.STATIC_DRAW);
    this.vao = vao;
    this.indexCount = idx.count;
    this.indexType = idx.type === 5125 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;

    const mat = glb.json.materials[0].pbrMetallicRoughness;
    this.rough = mat.roughnessFactor ?? 0.5;

    // base colour texture is embedded in the GLB
    const img = glb.json.images[0];
    const bv = glb.json.bufferViews[img.bufferView];
    const blob = new Blob([new Uint8Array(glb.bin, bv.byteOffset || 0, bv.byteLength)], { type: img.mimeType });
    const baseBmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    this.baseTex = this._texture(baseBmp, true);

    this.hasNrm = 0;
    if (normalUrl) {
      try {
        const nb = await (await fetch(normalUrl)).blob();
        const nbmp = await createImageBitmap(nb, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
        this.nrmTex = this._texture(nbmp, false);
        this.hasNrm = 1;
      } catch (e) { /* normal map is optional */ }
    }
    this.ready = true;
  }

  _texture(bitmap, srgb) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, srgb ? gl.SRGB8_ALPHA8 : gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const ext = gl.getExtension('EXT_texture_filter_anisotropic');
    if (ext) gl.texParameterf(gl.TEXTURE_2D, ext.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, gl.getParameter(ext.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
    return t;
  }

  resize() {
    const rect = this.glCanvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height));
    if (w === this.w && h === this.h && dpr === this.dpr) return;
    this.w = w; this.h = h; this.dpr = dpr;
    for (const c of [this.glCanvas, this.fx]) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
  }

  /** Camera distance so the ball fills `fraction` of the shorter screen side. */
  fitDistance(fraction) {
    const aspect = this.w / this.h;
    return (BALL_LENGTH / fraction) / (2 * Math.tan(FOV / 2) * Math.min(1, aspect));
  }

  project(p) {
    const c = transformPoint(this.vp, p);
    if (c[3] <= 0.01) return null;
    return [(c[0] / c[3] * 0.5 + 0.5) * this.w, (1 - (c[1] / c[3] * 0.5 + 0.5)) * this.h, c[3]];
  }

  /**
   * @param {object} s
   *  q: ball orientation in scene frame, pos: [x,y,z] metres, dist: camera distance, lookY: target y offset,
   *  ground: draw pitch, trail: [[x,y,z]...]
   */
  draw(s) {
    if (!this.gl) return;
    this.resize();
    const gl = this.gl;
    const aspect = this.w / this.h;
    const pos = s.pos || [0, 0, 0];
    const eye = [pos[0], pos[1] + (s.lookY || 0) * 0.35, pos[2] + s.dist];
    const tgt = [pos[0], pos[1] + (s.lookY || 0), pos[2]];
    const view = mat4LookAt(eye, tgt, [0, 1, 0]);
    const proj = mat4Perspective(FOV, aspect, 0.05, 400);
    const vp = mat4Mul(proj, view);
    this.vp = vp;

    // 2D layer (pitch, shadow, trail) sits under the ball
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    if (s.ground) this._drawGround(ctx, pos, eye, s.trail || []);

    gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.ready) return;
    gl.useProgram(this.prog);
    const model = mat4FromQuatPos(s.q, pos);
    const rot = mat3FromQuat(s.q);
    gl.uniformMatrix4fv(this.u.uModel, false, model);
    gl.uniformMatrix4fv(this.u.uViewProj, false, vp);
    gl.uniformMatrix3fv(this.u.uRot, false, rot);
    gl.uniform3f(this.u.uCam, eye[0], eye[1], eye[2]);
    gl.uniform1f(this.u.uRough, this.rough);
    gl.uniform1f(this.u.uHasNrm, this.hasNrm);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.baseTex); gl.uniform1i(this.u.uBase, 0);
    if (this.hasNrm) { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.nrmTex); }
    gl.uniform1i(this.u.uNrm, 1);
    gl.bindVertexArray(this.vao);
    gl.drawElements(gl.TRIANGLES, this.indexCount, this.indexType, 0);
  }

  _drawGround(ctx, pos, eye, trail) {
    const zNear = eye[2] - 0.6, zFar = -70;
    const x0 = pos[0] - 45, x1 = pos[0] + 45;
    const hor = this.project([pos[0], 0, -400]);
    const corners = [[x0, 0, zNear], [x1, 0, zNear], [x1, 0, zFar], [x0, 0, zFar]].map((p) => this.project(p));
    if (corners.every(Boolean) && hor) {
      const g = ctx.createLinearGradient(0, hor[1], 0, this.h);
      g.addColorStop(0, 'rgba(26,96,60,0)');
      g.addColorStop(0.35, 'rgba(26,96,60,0.55)');
      g.addColorStop(1, 'rgba(14,64,38,0.95)');
      ctx.fillStyle = g;
      ctx.beginPath();
      corners.forEach((c, i) => (i ? ctx.lineTo(c[0], c[1]) : ctx.moveTo(c[0], c[1])));
      ctx.closePath();
      ctx.fill();
    }
    // chalk lines every 5 m across the pitch, labelled with distance from the release point
    ctx.lineWidth = 1.5;
    ctx.font = '600 12px "Barlow Condensed", "Arial Narrow", sans-serif';
    ctx.textAlign = 'center';
    const first = Math.floor((pos[0] - 30) / 5) * 5;
    for (let x = first; x <= pos[0] + 60; x += 5) {
      const a = this.project([x, 0, zNear]);
      const b = this.project([x, 0, zFar]);
      if (!a || !b) continue;
      const major = Math.round(x) % 10 === 0;
      ctx.strokeStyle = major ? 'rgba(244,247,250,0.55)' : 'rgba(244,247,250,0.22)';
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      if (x >= 0 && major) {
        const lab = this.project([x, 0, eye[2] - 1.4]);
        if (lab && lab[1] < this.h - 4) {
          ctx.fillStyle = 'rgba(244,247,250,0.85)';
          ctx.fillText(`${x} m`, lab[0], Math.min(this.h - 8, lab[1]));
        }
      }
    }
    // shadow under the ball
    const sc = this.project([pos[0], 0, pos[2]]);
    if (sc) {
      const e = this.project([pos[0] + 0.22, 0, pos[2]]);
      const rx = e ? Math.abs(e[0] - sc[0]) : 10;
      const k = 1 / (1 + Math.max(0, pos[1]) * 0.5);
      ctx.fillStyle = `rgba(4,10,30,${0.38 * k})`;
      ctx.beginPath();
      ctx.ellipse(sc[0], sc[1], rx * (0.8 + 0.6 * k), rx * 0.22, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    // trail
    if (trail.length > 1) {
      ctx.lineCap = 'round';
      const n = trail.length;
      let prev = this.project(trail[0]);
      for (let i = 1; i < n; i++) {
        const cur = this.project(trail[i]);
        if (prev && cur) {
          const a = Math.pow(i / n, 1.6);
          ctx.strokeStyle = `rgba(255,255,255,${0.08 + 0.7 * a})`;
          ctx.lineWidth = 1 + 3.5 * a;
          ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(cur[0], cur[1]); ctx.stroke();
        }
        prev = cur;
      }
    }
  }
}
