# Third-party notices

Recipe-authored runtime code and documentation in this repository are
distributed under **AGPL-3.0-only**; see [LICENSE](LICENSE). This repository
does not redistribute any model weights. The components below remain under
their own licenses, which are not changed by this notice.

---

## Z.ai — GLM-5.3-Flash

- Model: https://huggingface.co/zai-org/GLM-5.3-Flash
- The base model and its weights are Z.ai's work under its own license. This
  repository serves a derivative quantized pack and the serving runtime; it
  names the model, credits the team, and redistributes nothing.
- This is independent community work, not an endorsement by Z.ai.

---

## Blackfrost-AI (Sir Frosty) — SAGE packs

- The served checkpoint `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` is a SAGE
  pack produced in official collaboration with Blackfrost-AI.
- **SAGE is proprietary.** This repository references the pack by name and bpw
  only: it does not document how the pack was produced, does not bundle any
  pack artifact, measurement, solver or calibration file, and does not link the
  weights. The pack stays behind a gated Hugging Face repository.

---

## Inco AI — DFlash2 draft model

- Model: https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- License: **CC BY-NC-ND 4.0 — non-commercial research and evaluation only.**
- Use here: unmodified speculative-decoding draft checkpoint, bind-mounted
  read-only at `/drafter`. This repository does not bundle or modify the
  DFlash2 weights. Commercial users must contact Inco AI for licensing.

---

## Turboderp — ExLlamaV3

https://github.com/turboderp-org/exllamav3

The EXL3 trellis format, codebooks and quantization method are ExLlamaV3's
work. This project serves packs in that format and calls its extension.

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## The vllm-exl3 plugin authors

The EXL3 serving plugin this runtime loads is the community's `vllm-exl3`
work; the loader patches in [patches/](patches/) adapt the installed snapshot
in place at boot and are published so the failure modes they fix are
documented. The plugin and vLLM remain under their own licenses.

---

## gitcommit90 — glm-5.3-one-spark

https://github.com/gitcommit90/glm-5.3-one-spark

The disjoint-slab drafter-pages mechanism re-derived in
[patches/gv_drafter_slabs.py](patches/gv_drafter_slabs.py) and the
indexer-workspace clamp in [patches/gv_indexer_ws.py](patches/gv_indexer_ws.py)
follow the design of their published work; the implementations here were
re-derived against this snapshot's planner and validated with the CPU-only
probe battery. Their deployment's receipts were used as the cross-check in the
prompt-parity measurements.

---

## Weschera — GLM-5.3-Flash-Unsloth-1x-DGX-Spark

Read-only reference deployment review; its published unified-memory failure
evidence (30-40 minute host wedges under allocation pressure) is the
independent confirmation behind [docs/SAFETY.md](docs/SAFETY.md).

---

## AgentSparko — experts-offload fork

`lane/guard.sh` is adapted from that recipe line's guard (MIT), with the swap
watcher and the sysctl save/restore protocol carried forward.

---

## vLLM

https://github.com/vllm-project/vllm — Apache-2.0. The engine snapshot this
recipe runs is pinned in the docker image named in the README; upstream PR
#60618 (fused `fused_mqa_topk`) is referenced as the identified prefill fix
and is not part of this repository.
