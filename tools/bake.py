"""Convert a Meshy-style rugby ball USDZ into the web assets used by Pass Lab.

Usage:
    unzip YourBall.usdz -d /tmp/ball_usdz
    python3 tools/bake.py /tmp/ball_usdz assets

It remeshes the (very dense) model into a clean ~25k-triangle mesh and bakes its colour and surface detail
into new seamless textures. Needs numpy, scipy and Pillow. Writes into the output folder:
  ball.glb            - low-poly mesh + baked base colour (glTF 2.0, metres, long axis = +Y)
  ball_normal_os.jpg  - object-space normal map (baked from the high-res geometry + original normal map)
Assumes the ball's long axis is the X axis of the source model (as Meshy exports it).
"""
import sys, io, json, struct, time, glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from usdc import Crate

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC = sys.argv[1].rstrip('/') + '/'
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
NTH, NPH = 128, 96            # grid segments around / along
TEX_W, TEX_H = 2048, 2048     # baked base colour
NRM_W, NRM_H = 1024, 1024     # baked object-space normals
BALL_LEN_M = 0.29             # real size-5 ball ~ 280-300 mm

t0 = time.time()
usdc_files = glob.glob(SRC + '*.usdc')
if not usdc_files:
    sys.exit('No .usdc file found in ' + SRC)
c = Crate(open(usdc_files[0], 'rb').read())
tex = {}
for path, st, fields in c.specs:
    if path.endswith('.inputs:file'):
        d = dict(fields)
        if 'default' in d:
            tex[path.split('/')[-1].split('.')[0]] = c.value(d['default'])
base_name = tex.get('tex_base_color', './extracted_image_0.jpg')
normal_name = tex.get('tex_normal', './extracted_image_2.jpg')
g = {}
for path, st, fields in c.specs:
    if path.startswith('/Root/mesh1.'):
        d = dict(fields)
        if 'default' in d:
            g[path.split('.')[1]] = c.value(d['default'])
P = g['points'].astype(np.float64)
N = g['normals'].astype(np.float64)
UV = g['primvars:st'].astype(np.float64)
TAN = g['primvars:tangents'].astype(np.float64)
F = g['faceVertexIndices'].reshape(-1, 3)
print('loaded', P.shape, F.shape, '%.1fs' % (time.time() - t0))

# rotate +90deg about Z so the long axis (X) becomes +Y:  (x,y,z) -> (-y, x, z)
def rot(v):
    return np.stack([-v[:, 1], v[:, 0], v[:, 2]], axis=1)
P, N = rot(P), rot(N)
TAN = np.concatenate([rot(TAN[:, :3]), TAN[:, 3:]], axis=1)
ctr = (P.min(0) + P.max(0)) / 2
P -= ctr
print('bounds', P.min(0), P.max(0))

# ---------- radial surface table r(phi, theta) around the centre ----------
rho = np.linalg.norm(P, axis=1)
dirs = P / rho[:, None]
phi = np.arccos(np.clip(dirs[:, 1], -1, 1))             # 0 at +Y tip
theta = np.mod(np.arctan2(dirs[:, 2], dirs[:, 0]), 2 * np.pi)

# fine table, then smooth, then sampled at grid nodes / texels
TH_T, PH_T = 256, 192
ti = np.minimum((theta / (2 * np.pi) * TH_T).astype(int), TH_T - 1)
pi_ = np.minimum((phi / np.pi * PH_T).astype(int), PH_T - 1)
sums = np.zeros((PH_T, TH_T)); cnt = np.zeros((PH_T, TH_T))
np.add.at(sums, (pi_, ti), rho)
np.add.at(cnt, (pi_, ti), 1)
with np.errstate(invalid='ignore'):
    table = sums / cnt
# fill holes by iterative neighbour average
for _ in range(50):
    holes = np.isnan(table)
    if not holes.any():
        break
    pad = np.pad(table, ((1, 1), (0, 0)), mode='edge')
    nb = np.stack([pad[:-2], pad[2:], np.roll(table, 1, 1), np.roll(table, -1, 1)])
    fill = np.nanmean(np.where(np.isnan(nb), np.nan, nb), axis=0)
    table = np.where(holes, fill, table)
# light smoothing (3x3)
def smooth(t):
    pad = np.pad(t, ((1, 1), (0, 0)), mode='edge')
    acc = np.zeros_like(t)
    for dy in (0, 1, 2):
        for dx in (-1, 0, 1):
            acc += np.roll(pad[dy:dy + t.shape[0]], dx, axis=1)
    return acc / 9
for _ in range(2):
    table = smooth(table)
print('radial table', table.min(), table.max(), '%.1fs' % (time.time() - t0))

def radius(ph, th):
    """bilinear sample of the radial table, theta periodic."""
    fy = np.clip(ph / np.pi * PH_T - 0.5, 0, PH_T - 1)
    fx = th / (2 * np.pi) * TH_T - 0.5
    y0 = np.floor(fy).astype(int); y1 = np.minimum(y0 + 1, PH_T - 1); wy = fy - y0
    x0 = np.floor(fx).astype(int); wx = fx - x0
    x0m = np.mod(x0, TH_T); x1m = np.mod(x0 + 1, TH_T)
    return ((table[y0, x0m] * (1 - wx) + table[y0, x1m] * wx) * (1 - wy)
            + (table[y1, x0m] * (1 - wx) + table[y1, x1m] * wx) * wy)

def surface(ph, th):
    r = radius(ph, th)
    return np.stack([r * np.sin(ph) * np.cos(th), r * np.cos(ph), r * np.sin(ph) * np.sin(th)], axis=-1)

# ---------- low-poly grid ----------
js, is_ = np.meshgrid(np.arange(NTH + 1), np.arange(NPH + 1))
th_g = js / NTH * 2 * np.pi
ph_g = is_ / NPH * np.pi
GP = surface(ph_g, th_g)                                  # (NPH+1, NTH+1, 3)
# smooth normals from finite differences of the analytic surface
eps = 1e-3
dth = (surface(ph_g, th_g + eps) - surface(ph_g, th_g - eps))
dph = (surface(np.clip(ph_g + eps, 0, np.pi), th_g) - surface(np.clip(ph_g - eps, 0, np.pi), th_g))
GN = np.cross(dph, dth)
ln = np.linalg.norm(GN, axis=-1, keepdims=True)
GN = np.where(ln > 1e-12, GN / np.maximum(ln, 1e-12), GP / np.linalg.norm(GP, axis=-1, keepdims=True))
# make sure normals point outward
flip = (np.sum(GN * GP, axis=-1) < 0)[..., None]
GN = np.where(flip, -GN, GN)
# fix poles: normal = +-Y
GN[0] = np.array([0, 1, 0]); GN[-1] = np.array([0, -1, 0])
GUV = np.stack([js / NTH, (is_ + 0.0) / NPH], axis=-1)

verts = GP.reshape(-1, 3); norms = GN.reshape(-1, 3); uvs = GUV.reshape(-1, 2)
idx = []
for i in range(NPH):
    for j in range(NTH):
        a = i * (NTH + 1) + j; b = a + 1; cc = a + (NTH + 1); d = cc + 1
        # winding chosen so geometric normal points outward; verified below
        idx += [a, cc, b, b, cc, d]
idx = np.array(idx, dtype=np.uint32).reshape(-1, 3)
# verify winding outward
v0, v1, v2 = verts[idx[:, 0]], verts[idx[:, 1]], verts[idx[:, 2]]
fn = np.cross(v1 - v0, v2 - v0)
cen = (v0 + v1 + v2) / 3
good = np.sum(fn * cen, axis=1)
valid = np.linalg.norm(fn, axis=1) > 1e-12
if (good[valid] < 0).mean() > 0.5:
    idx = idx[:, [0, 2, 1]]
    print('flipped winding')
idx = idx[np.linalg.norm(np.cross(verts[idx[:, 1]] - verts[idx[:, 0]], verts[idx[:, 2]] - verts[idx[:, 0]]), axis=1) > 1e-14]
print('low-poly', verts.shape, idx.shape)

# ---------- bake ----------
tri_c = (P[F[:, 0]] + P[F[:, 1]] + P[F[:, 2]]) / 3
tree = cKDTree(tri_c)
imgc = np.asarray(Image.open(SRC + base_name.lstrip('./')).convert('RGB')).astype(np.float32)
imgn = np.asarray(Image.open(SRC + normal_name.lstrip('./')).convert('RGB')).astype(np.float32)

def bake(W, H, want_color, want_normal):
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    th = (px + 0.5) / W * 2 * np.pi
    ph = (py + 0.5) / H * np.pi
    pts = surface(ph, th).reshape(-1, 3)
    # keep the same texel->surface mapping as the mesh UVs: u = theta/2pi, v = phi/pi
    _, tid = tree.query(pts, k=1, workers=-1)
    tri = F[tid]
    a, b, cc = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    v0, v1, v2 = b - a, cc - a, pts - a
    d00 = np.einsum('ij,ij->i', v0, v0); d01 = np.einsum('ij,ij->i', v0, v1); d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0); d21 = np.einsum('ij,ij->i', v2, v1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    w0 = 1 - w1 - w2
    W3 = np.clip(np.stack([w0, w1, w2], axis=1), 0, 1.2)   # allow slight extrapolation, clamp wild values
    W3 = np.clip(W3, -0.1, 1.1)
    W3 /= W3.sum(1, keepdims=True)
    def interp(arr):
        return (arr[tri[:, 0]] * W3[:, 0:1] + arr[tri[:, 1]] * W3[:, 1:2] + arr[tri[:, 2]] * W3[:, 2:3])
    uv = interp(UV)
    # sample original textures (USD uv origin bottom-left)
    sx = uv[:, 0] * imgc.shape[1] - 0.5
    sy = (1 - uv[:, 1]) * imgc.shape[0] - 0.5
    out_c = out_n = None
    if want_color:
        chans = [map_coordinates(imgc[..., k], [sy, sx], order=1, mode='wrap') for k in range(3)]
        out_c = np.clip(np.stack(chans, axis=1), 0, 255).astype(np.uint8).reshape(H, W, 3)
    if want_normal:
        sxn = uv[:, 0] * imgn.shape[1] - 0.5
        syn = (1 - uv[:, 1]) * imgn.shape[0] - 0.5
        nt = np.stack([map_coordinates(imgn[..., k], [syn, sxn], order=1, mode='wrap') for k in range(3)], axis=1) / 255.0
        nt = nt * 2 - 1
        Nn = interp(N); Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
        T = interp(TAN[:, :3]); T -= Nn * np.sum(T * Nn, axis=1, keepdims=True)
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        sgn = interp(TAN[:, 3:4])
        sgn = np.where(sgn < 0, -1.0, 1.0)
        B = np.cross(Nn, T) * sgn
        nobj = T * nt[:, 0:1] + B * nt[:, 1:2] + Nn * nt[:, 2:3]
        nobj /= np.maximum(np.linalg.norm(nobj, axis=1, keepdims=True), 1e-9)
        out_n = np.clip((nobj * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8).reshape(H, W, 3)
        # also report how much the baked normal differs from the smooth analytic one
    return out_c, out_n

col, _ = bake(TEX_W, TEX_H, True, False)
print('baked colour %.1fs' % (time.time() - t0))
_, nrm = bake(NRM_W, NRM_H, False, True)
print('baked normals %.1fs' % (time.time() - t0))

# fill poles' degenerate rows with the average of the neighbouring row so the tip does not show swirl
def pole_fix(img, rows=3):
    img = img.copy()
    for r in range(rows):
        mean_top = img[rows].mean(axis=0, keepdims=True)
        img[r] = mean_top
        mean_bot = img[-rows - 1].mean(axis=0, keepdims=True)
        img[-r - 1] = mean_bot
    return img

# ---------- scale to metres and write ----------
scale = BALL_LEN_M / (verts[:, 1].max() - verts[:, 1].min())
verts_m = (verts * scale).astype(np.float32)
bio = io.BytesIO()
Image.fromarray(col).save(bio, 'JPEG', quality=86, optimize=True, progressive=True)
col_jpg = bio.getvalue()
Image.fromarray(nrm).save(OUT + '/ball_normal_os.jpg', 'JPEG', quality=92, optimize=True)

pos_b = verts_m.astype('<f4').tobytes()
nor_b = norms.astype('<f4').tobytes()
uv_b = uvs.astype('<f4').tobytes()
idx_b = idx.astype('<u2').ravel().tobytes() if verts.shape[0] < 65536 else idx.astype('<u4').ravel().tobytes()
idx_type = 5123 if verts.shape[0] < 65536 else 5125

def pad4(b, fill=b'\0'):
    return b + fill * ((4 - len(b) % 4) % 4)

chunks = []
views = []
def add_view(data, target=None):
    off = sum(len(pad4(x)) for x in chunks)
    chunks.append(data)
    v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
    if target:
        v['target'] = target
    views.append(v)
    return len(views) - 1

vp = add_view(pos_b, 34962); vn = add_view(nor_b, 34962); vu = add_view(uv_b, 34962)
vi = add_view(idx_b, 34963); vimg = add_view(col_jpg)
accessors = [
    {'bufferView': vp, 'componentType': 5126, 'count': len(verts_m), 'type': 'VEC3',
     'min': verts_m.min(0).tolist(), 'max': verts_m.max(0).tolist()},
    {'bufferView': vn, 'componentType': 5126, 'count': len(norms), 'type': 'VEC3'},
    {'bufferView': vu, 'componentType': 5126, 'count': len(uvs), 'type': 'VEC2'},
    {'bufferView': vi, 'componentType': idx_type, 'count': int(idx.size), 'type': 'SCALAR'},
]
gltf = {
    'asset': {'version': '2.0', 'generator': 'rugby-pass-lab bake script'},
    'scene': 0, 'scenes': [{'nodes': [0]}],
    'nodes': [{'mesh': 0, 'name': 'EnglandRugbyBall'}],
    'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2}, 'indices': 3, 'material': 0}]}],
    'materials': [{'name': 'ball', 'pbrMetallicRoughness': {
        'baseColorTexture': {'index': 0}, 'metallicFactor': 0.0, 'roughnessFactor': 0.52},
        'extras': {'objectSpaceNormalMap': 'ball_normal_os.jpg'}}],
    'textures': [{'source': 0, 'sampler': 0}],
    'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 33071}],
    'images': [{'bufferView': vimg, 'mimeType': 'image/jpeg'}],
    'accessors': accessors, 'bufferViews': views,
    'buffers': [{'byteLength': sum(len(pad4(x)) for x in chunks)}],
}
jb = pad4(json.dumps(gltf, separators=(',', ':')).encode(), b' ')
bb = b''.join(pad4(x) for x in chunks)
glb = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(bb))
glb += struct.pack('<II', len(jb), 0x4E4F534A) + jb + struct.pack('<II', len(bb), 0x004E4942) + bb
open(OUT + '/ball.glb', 'wb').write(glb)
print('wrote ball.glb %.2f MB, normal %.0f KB, %.1fs' % (len(glb) / 1e6, len(open(OUT + '/ball_normal_os.jpg', 'rb').read()) / 1e3, time.time() - t0))
