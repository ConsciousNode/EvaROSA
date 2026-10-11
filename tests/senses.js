// Senses: finite-difference check of the paired path [K sense prefixes] + caption, through the
// SenseAdapter and the whole RWKV, plus feature-extractor sanity. `node tests/senses.js`
const fs = require('fs'), path = require('path'), vm = require('vm');
const src = fs.readFileSync(path.join(__dirname, '..', process.env.MODEL_FILE || 'evarosa.html'), 'utf8');
const cut = (a, b) => { const i = src.indexOf(a), j = src.indexOf(b, i); if (i < 0 || j < 0) throw new Error('marker ' + a); return src.slice(i, j); };
const lmEnd = src.indexOf('/* ╔', src.indexOf('class LanguageModel {'));
const code = [cut('const CONFIG = {', '/* ╔'), cut('class TMAC {', 'class AdamOptimizer'), cut('function initPoSTDecayGates', '/* ╔'),
  src.slice(src.indexOf('class RWKVv7Block {'), lmEnd), cut('const SENSE_K', '/* ╔'), cut('class Trainer {', '/* ╔'),
  'this.M = { BitLinear, LanguageModel, SenseAdapter, SenseFeatures, Trainer };'].join('\n');
const noop = () => 0;
const ctx = { setInterval: noop, clearInterval: noop, setTimeout: noop, document: undefined, Math, Float32Array: Float64Array, Int8Array, Uint8Array, Array, Object, console };
vm.createContext(ctx); vm.runInContext(code, ctx);
const M = ctx.M; let seed = 7;
ctx.Math.random = Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
M.BitLinear.prototype.forward = function (x) { const o = new Float64Array(this.outDim); for (let r = 0; r < this.outDim; r++) { let a = 0; for (let i = 0; i < this.inDim; i++) a += this.weight[r * this.inDim + i] * x[i]; o[r] = a; } return o; };
M.BitLinear.prototype.backward = function (g, x) { const gi = new Float64Array(this.inDim); for (let r = 0; r < this.outDim; r++) { if (!g[r]) continue; for (let i = 0; i < this.inDim; i++) { this.grad[r * this.inDim + i] += g[r] * x[i]; gi[i] += this.weight[r * this.inDim + i] * g[r]; } } return gi; };   // float STE pair
let fails = 0;

// ── 1. feature extractors ─────────────────────────────────────────────
const img = (fn, W = 40, H = 30) => { const d = new Uint8Array(W * H * 4); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const [r, g, b] = fn(x / W, y / H); const p = (y * W + x) * 4; d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 255; } return { data: d, width: W, height: H }; };
const fRed = M.SenseFeatures.image(img(() => [255, 0, 0])), fSplit = M.SenseFeatures.image(img(x => x < 0.5 ? [0, 0, 0] : [255, 255, 255]));
const ok1 = fRed.length === 272 && Math.abs(fRed[0] - 1) < 1e-9 && Math.abs(fRed[1] + 1) < 1e-9 && fRed[192] === 0 && fSplit[192 + 3] > 0.5;
console.log(`image features: red cell = [${fRed[0]}, ${fRed[1]}, ${fRed[2]}], flat edges ${fRed[192]}, split-image edge ${fSplit[192 + 3].toFixed(2)}  ${ok1 ? 'ok' : 'FAIL'}`); if (!ok1) fails++;
const sr = 16000, tone = f => Float32Array.from({ length: sr }, (_, i) => 0.5 * Math.sin(2 * Math.PI * f * i / sr));
const aLo = M.SenseFeatures.audio(tone(200), sr), aHi = M.SenseFeatures.audio(tone(3000), sr), aSil = M.SenseFeatures.audio(new Float32Array(sr), sr);
const peak = a => { let b = 0; for (let i = 0; i < 24; i++) if (a[i] > a[b]) b = i; return b; };
const ok2 = aLo.length === 64 && peak(aLo) < peak(aHi) && aLo.every(Number.isFinite) && aSil.every(Number.isFinite) && aHi[52] > aLo[52];
console.log(`audio features: 200 Hz peaks in band ${peak(aLo)}, 3 kHz in band ${peak(aHi)}, brightness ${aLo[52].toFixed(2)} < ${aHi[52].toFixed(2)}, silence finite  ${ok2 ? 'ok' : 'FAIL'}`); if (!ok2) fails++;

// ── 2. gradient check of the paired path ──────────────────────────────
const cfg = { vocab: 20, dim: 12, layers: 2 };
const m = new M.LanguageModel(cfg);
for (const b of m.blocks) { for (let i = 0; i < b.W_x_w.weight.length; i++) b.W_x_w.weight[i] *= 0.3; b.mu_r.fill(0.4); b.mu_k.fill(0.6); }
const ad = new M.SenseAdapter('vision', 10, cfg.dim, 3);
for (let i = 0; i < ad.C.length; i++) ad.C[i] = Math.sin(i * 0.37) * 0.2;   // non-zero so the context path is exercised
const feats = Float64Array.from({ length: 10 }, (_, i) => Math.sin(i * 1.7) * 0.8), caption = [4, 11, 2, 17];
for (let n = 0; n < 30; n++) ad.observe(Float64Array.from({ length: 10 }, () => Math.random() * 2 - 1));   // warm the running stats (one sample alone standardizes to 0)
const tr = new M.Trainer(m, null);
const ce = (lg, t) => { let mx = -1e9; for (const v of lg) mx = Math.max(mx, v); let s = 0; for (const v of lg) s += Math.exp(v - mx); return -(lg[t] - mx - Math.log(s)); };
const loss = () => { const pre = ad.prefix(feats), cv = ad.context(feats); let st = m.getEmptyState(), r, L = 0;
  for (const v of pre) { r = m.forwardVec(v, st); st = r.state; }
  for (let t = 0; t < caption.length; t++) { L += ce(r.logits, caption[t]); if (t < caption.length - 1) { r = m.forward(caption[t], st, cv); st = r.state; } } return L; };
m.zeroGrad(); m.embGrad.fill(0); ad.zeroGrad();
const Lan = tr.trainPaired(ad, feats, caption);
const Lfd = loss() / caption.length, ok3 = Math.abs(Lan - Lfd) < 1e-5;   // trainer adds 1e-8 inside its log
console.log(`trainPaired loss ${Lan.toFixed(6)} vs direct ${Lfd.toFixed(6)}  ${ok3 ? 'ok' : 'FAIL'}`); if (!ok3) fails++;
const eps = 1e-6;
function check(label, W, G) {
  let dot = 0, na = 0, nn = 0;
  for (let i = 0; i < W.length; i++) { const o = W[i]; W[i] = o + eps; const a = loss(); W[i] = o - eps; const b = loss(); W[i] = o; const num = (a - b) / (2 * eps); dot += G[i] * num; na += G[i] * G[i]; nn += num * num; }
  if (nn < 1e-14) { console.log(`${label.padEnd(20)} (unused)`); return; }
  const cos = dot / Math.sqrt(na * nn), ratio = Math.sqrt(na / nn), ok = cos > 0.9999 && Math.abs(ratio - 1) < 1e-3; if (!ok) fails++;
  console.log(`${label.padEnd(20)} cos=${cos.toFixed(6)} ratio=${ratio.toFixed(6)} ${ok ? 'ok' : 'FAIL'}`);
}
check('adapter.W', ad.W, ad.gW);
check('adapter.b', ad.b, ad.gb);
check('adapter.C (context)', ad.C, ad.gC);
check('embedding', m.embedding, m.embGrad);
check('block0.W_o', m.blocks[0].W_o.weight, m.blocks[0].W_o.grad);
check('block1.W_x_k', m.blocks[1].W_x_k.weight, m.blocks[1].W_x_k.grad);
check('head', m.head.weight, m.head.grad);
console.log(fails ? `${fails} FAILED` : 'ALL OK — senses path is exact');
process.exit(fails ? 1 : 0);
