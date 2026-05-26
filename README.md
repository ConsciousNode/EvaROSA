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
