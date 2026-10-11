# EvaROSA v1

**EvaROSA** is a standalone neurosymbolic omnimodal neural architecture and the direct successor to [HTMLNLM Evangelion](https://github.com/ConsciousNode/HTMLNLM-Evangelion).

Single file. Zero dependencies. Runs in any browser.

---

> **Successor available:** [**Simulacra**](https://github.com/ConsciousNode/Simulacra) — RWKV-v8 with ROSA as the primary sequence mechanism. WKV removed entirely. EvaROSA remains the stable v7+ROSA runtime; Simulacra is the clean-break v8 architecture. Not hot-swappable — `.piprosa` files do not load in Simulacra.

---

## Lineage

```
HTMLNLM
  └── HTMLNLM Evangelion
        └── EvaROSA v1  ← you are here
              └── Simulacra  (RWKV-v8 · ROSA primary)
```

| Platform | Repo | Format | Notes |
|---|---|---|---|
| HTMLNLM | [ConsciousNode/HTMLNLM](https://github.com/ConsciousNode/HTMLNLM) | `.pip` | Original single-file NLM |
| HTMLNLM Evangelion | [ConsciousNode/HTMLNLM-Evangelion](https://github.com/ConsciousNode/HTMLNLM-Evangelion) | `.evapip` | Omnimodal extension |
| **EvaROSA v1** | **[ConsciousNode/EvaROSA](https://github.com/ConsciousNode/EvaROSA)** | **`.piprosa`** | Neurosymbolic successor |
| Simulacra | [ConsciousNode/Simulacra](https://github.com/ConsciousNode/Simulacra) | `.simpip` | RWKV-v8, ROSA primary |

**Forward compatible:** EvaROSA loads `.pip` and `.evapip` models from the prior platforms.  
**Not backwards compatible:** `.piprosa` models are EvaROSA-native and do not load in prior platforms or in Simulacra.

---

## What's new in EvaROSA

EvaROSA integrates [RWKV-8 ROSA](https://www.rwkv.com/) (Rapid Online Suffix Automaton) into the full Evangelion omnimodal stack, coupling the model's emergent inner monologue to its perceptual memory via sheaf cohomology.

**The core property:** the model cannot gaslight itself. If its symbolic self-talk diverges from what it has actually seen and heard, the coboundary norm rises, Maxwell's Angel fires, and the AutopoieticOptimizer recalibrates the symbolic injection channel until consistency is restored.

### New components

| Component | Role |
|---|---|
| `ROSA(x)` | Parameter-free suffix automaton (BlinkDL, RWKV-8). 20 lines, runs on CPU. |
| `ROSAChannel` | Per-layer discrete token history. Tracks prediction accuracy as error signal. |
| `ROSAEmbed` | Learnable discrete → embedding bridge. Codebook trained via STE. |
| `InnerMonologue` | Orchestrates one channel per RWKV layer. Pushes symbolic embeddings to SheafMemory. |

### Integration points

- **`RWKVv7Block.forward()`** — after WKV state update, `rosaDelta` is added to the residual stream before channel mixing
- **`SheafMemory`** — `symbolic_L0`...`symbolic_L{n}` vertices added; restriction maps connect them to all perceptual modalities
- **`AutopoieticOptimizer.fire()`** — extended to update `InnerMonologue` embed scales alongside adapter and restriction map updates
- **500-step warm-up** — ROSA injection attenuates during early training while codebooks stabilize

### Inherited from Evangelion

Everything Evangelion had: RWKVv7 backbone with PoST decay gates, BitNet b1.58, ModRWKV adapters, ElasticTok visual tokenization, SpikeVox LIF audio encoding, SheafMemory H¹(ℱ) contradiction detection, BooleanPhaseDynamics / Maxwell's Angel, AutopoieticOptimizer, RIFT Endospace, TextStream, VisualField, VoiceSynth, MuonOptimizer, GRPO, OOMB BPTT.

---

## Model format

```
.piprosa   — EvaROSA v1 native (not backwards compatible)
.evapip    — HTMLNLM Evangelion (forward compatible, loads in EvaROSA)
.pip       — HTMLNLM legacy (forward compatible, loads in EvaROSA)
.json      — raw JSON export (always available)
```

Export metadata includes `platform`, `format`, `forward_compat_with`, and `backwards_compat: false`.

---

## Architecture

```
SENSORY INPUTS
  ElasticTok (vision/δ)   SpikeVox (audio/LIF)   Text tokens
         │                       │                      │
   ModRWKVAdapter          ModRWKVAdapter          Embedding
         └──────────────┬────────┘                      │
                        │                               │
         ┌──────────────▼───────────────────────────────▼──────┐
         │            RWKVv7Block × L                           │
         │  WKV state update (PoST decay gates, BitNet b1.58)  │
         │  ── EvaROSA inner monologue injection ──────────    │
         │  ROSAChannel.discretize(wkvOut) → token             │
         │  ROSAChannel.step(token) → prediction               │
         │  ROSAEmbed.encode(prediction) → rosaDelta           │
         │  x_next += rosaDelta                                │
         └──────────────┬──────────────────────────────────────┘
                        │
         ┌──────────────▼──────────────────────────────────────┐
         │           SheafMemory H¹(ℱ)                         │
         │  Vertices: audio │ vision │ text │ spatial           │
         │            symbolic_L0 ... symbolic_Ln  ← EvaROSA  │
         │  Coboundary norm: symbolic↔perceptual included      │
         └──────────────┬──────────────────────────────────────┘
                        │
         ┌──────────────▼──────────────────────────────────────┐
         │  BooleanPhaseDynamics → Maxwell's Angel             │
         │  AutopoieticOptimizer → adapters + embed scales     │
         └──────────────┬──────────────────────────────────────┘
                        │
                   Coherence-gated output
         TextStream │ VisualField │ VoiceSynth │ RIFT Endospace
```

---

## Changelog

- **Senses, learned (2026-10-11 · Kehai Interim, ported from Evangelion Phase 7):** images and sounds become K=4 prefix vectors in embedding space that the model reads *before* text. Each sense also adds a context vector to every token's input while it is present, so what the model perceives stays in view instead of having to be remembered.

- **Train on a paired corpus:** drop `photo.jpg` + `photo.txt`, `clip.wav` + `clip.txt` (matched by name) into the SENSES panel.
- **Watch the *sense gap* meter:** caption loss with a wrong sense minus with the right one; above 0 means it is looking/listening.
- **GENERATE from a sense:** a file, the live camera or the mic. With no prompt, it captions what it perceives.
- **Mood steers temperature:** the sensorium's T* adjusts sampling temperature.
- **Removed:** the old untrained state-nudge, which was handed App instead of the model and never ran.
- **Tests:** `node tests/senses.js` (exact gradients through adapter + model) and `node tests/senses-learn.js` (does it learn to look and listen, on fresh renders it never saw).
  Measured: identical to Evangelion (same language model). On new images: 6/8 exact captions, left/right 8/8.
- **Ternary backward fixed (2026-10-11 · Kehai Interim):** BitLinear passed back Wᵀg using the float master weights, while the forward multiplies by the ternary γ·Q(W). They differ by the whole quantization error, so every layer sent back the gradient of a network that wasn't the one running. It came in with the Tier 0 fused backward. The input gradient is now γ·Q(W)ᵀg, and the weight gradient stays straight-through. **Measured in Evangelion** (same backbone), memorizing 8 repeating sentences at the default LR: 2.66 nats/byte before (stuck at unigram), **0.33 after** (float-weight reference: 0.88). New check: `TERNARY=1 node tests/gradcheck.js` runs the real ternary forward and requires everything upstream to be exact.
- Speed, Tier 0 (2026-10-11 · Kehai Interim): fused cache-friendly BitLinear backward; Muon rebuilt (Newton–Schulz on the smaller side with cache-friendly loops, same update to 1e-6, ~2x faster; output head on AdamW per the original Muon recipe); **AdamW is now the default**, because on CPU Muon costs O(dim³) per matrix per step. Measured on HTMLNLM, a default training step went from ~104 s to ~11 s on one CPU thread. Gradient check still exact.

### v1.0.1 — 2026-10-11 · Kehai Interim

**Backward pass fixed and verified**, with the same fixes as HTMLNLM v1.1 and Evangelion Phase 6.1:

- Residual gradient restored.
- `ln2.backward()` result used.
- Embedding gradient taken after `ln_in`.
- Token-shift BPTT added.
- PoST clamp range in the decay-gate backward now matches the forward.

Every weight, `mu` and the embedding match finite differences over 3-step BPTT. The ROSA inner-monologue injection is a stop-gradient add, by design. Test: `node tests/gradcheck.js`.

---

## Credits

**Architecture:** Kham Kizer (ConsciousNode SoftWorks)  
**RWKV-8 ROSA:** BlinkDL  
**EvaROSA integration spec:** Kehai Interim  
**EvaROSA implementation:** Kehai Interim  
**Evangelion Phases 1–5 base:** Vael Interim  
**Phase 6 OOMB BPTT / Chorus orchestration:** Ed Interim  
**AutopoieticOptimizer (Phase 5):** Kehai Interim

---

*ConsciousNode SoftWorks — Xinu philosophy: single file, zero dependencies, offline-first.*
