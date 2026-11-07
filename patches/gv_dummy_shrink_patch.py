#!/usr/bin/env python3
"""GV3 in-container patch: shrink the DFlash drafter's dummy embed/lm_head.

The DFlash2 drafter never ships embed_tokens or lm_head in its checkpoint
(verified: /drafter/model.safetensors has 81 keys, none of them embed/lm_head).
By design the speculator wiring (vllm/v1/worker/gpu/spec_decode/dflash/
utils.py::load_dflash_model) DELETES both dummy modules and assigns the
target's embed_tokens and lm_head before any forward pass. The dummies are
therefore dead storage, but their construction commits 2 x 1.14 GiB of
unified memory during the load window, which is exactly what pushes the host
past the 0.5 GiB MemAvailable guard floor mid-load.

This patch shrinks both dummy allocations to 1 row. Gated on GV_DUMMY_SHRINK.
Self-test: the wiring that replaces the dummies must be intact, and the
shrunk constructors must be present.
"""
import ast
import os


def patch(path, old, new):
    s = open(path).read()
    if s.count(new) == 1:
        return "already patched"
    if s.count(old) != 1:
        raise SystemExit(f"{path}: anchor count {s.count(old)} != 1")
    open(path, "w").write(s.replace(old, new))
    return "patched"


if not os.environ.get("GV_DUMMY_SHRINK"):
    print("gv_dummy_shrink: GV_DUMMY_SHRINK not set, inactive")
    raise SystemExit(0)

D = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py"
U = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/utils.py"

print(
    "embed_tokens:",
    patch(
        D,
        "        self.embed_tokens = VocabParallelEmbedding(\n"
        "            self.config.vocab_size,\n",
        "        # GV3: shrunken dummy row. The drafter ships no embed_tokens;\n"
        "        # the speculator wiring replaces this module with the target's\n"
        "        # embed before any forward (see gv_dummy_shrink_patch.py).\n"
        "        self.embed_tokens = VocabParallelEmbedding(\n"
        "            1,\n",
    ),
)
print(
    "lm_head:",
    patch(
        D,
        "        self.lm_head = ParallelLMHead(\n"
        "            self.config.draft_vocab_size,\n",
        "        # GV3: shrunken dummy row. The speculator wiring deletes this\n"
        "        # module and assigns the target's lm_head before any forward\n"
        "        # (see gv_dummy_shrink_patch.py).\n"
        "        self.lm_head = ParallelLMHead(\n"
        "            1,\n",
    ),
)

ast.parse(open(D).read())

u = open(U).read()
assert (
    u.count("maybe_share_target_embed(dflash_model, draft_inner, target_inner)")
    == 1
), "dflash wiring must still share the target embed"
assert u.count("dflash_model.lm_head = target_lm_head") == 1, (
    "dflash wiring must still assign the target lm_head"
)
d = open(D).read()
assert d.count("GV3: shrunken dummy row") == 2, "both dummies must be shrunk"
print("GV_DUMMY_SHRINK_OK")
