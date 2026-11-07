#!/usr/bin/env python3
"""LANE GV container boot patch (idempotent, in-container, run before vllm serve).

1. vllm glm5next/common/kda.py: keep the quant config around the KDA attention
   construction when the pack declares non_routed_exl3 dense EXL3 tensors
   (adapted from the K2 recipe's patch_glm53_dense_exl3_quant_config.py).
2. vllm glm5next/common/model.py: same guard for the MLA attention.
3. installed vllm_exl3/exl3.py: extend the non_routed_exl3 bits allowlist to
   K7/K8 (the non-routed path builds per-shard exllamav3 LinearEXL3 instances,
   which support K2..K8; the 2..6 bound was policy, not a kernel limit).
4. installed vllm_exl3/exl3.py: skip 128-padding for bf16_shards rows
   (682b3ca) — pad only the EXL3 shards in a merged module.
5. installed vllm_exl3/exl3.py: strip the VL wrapper prefix in
   _bf16_shards_for (89e442d) — create_weights queries with the raw
   layer.prefix (language_model.model.layers.N...), which must be normalized
   the same way get_quant_method already does before the exact layers-dict
   lookup.
6. self-test: the exact lookups the GLM-5.3 overlay needs (KDA in_proj
   bf16_shards + per-layer bits) must resolve, else abort before serving.
"""
import sys

def patch(path, old, new):
    s = open(path).read()
    if s.count(new) == 1:
        return "already patched"
    if s.count(old) != 1:
        raise SystemExit(f"{path}: anchor count {s.count(old)} != 1")
    open(path, "w").write(s.replace(old, new))
    return "patched"

V = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common"
E = "/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py"

print("kda.py:", patch(
    f"{V}/kda.py",
    "        saved_quant_config = vllm_config.quant_config\n"
    "        try:\n"
    "            vllm_config.quant_config = None\n",
    "        saved_quant_config = vllm_config.quant_config\n"
    "        try:\n"
    "            # Keep the quant config when the pack declares dense EXL3 tensors\n"
    "            # for these projections (vllm-exl3 non_routed_exl3); unmatched\n"
    "            # linears still resolve to the unquantized method.\n"
    "            if not getattr(saved_quant_config, \"non_routed_exl3\", None):\n"
    "                vllm_config.quant_config = None\n",
))
print("model.py:", patch(
    f"{V}/model.py",
    "                quant_config=None,  # MLA projections are BF16 in checkpoint\n",
    "                # MLA projections are BF16 unless the pack declares dense EXL3\n"
    "                quant_config=(\n"
    "                    quant_config\n"
    "                    if getattr(quant_config, \"non_routed_exl3\", None)\n"
    "                    else None\n"
    "                ),\n",
))
for old, new in (
    # torch 2.14 regression fix #2: getattr(torch.ops, "_exl3_ops_ready", False)
    # returns an auto-created truthy module object instead of raising, so
    # _EXL3_OPS_READY is True at import and _exl3_register_custom_ops skips
    # registration while apply() still dispatches to the missing op. Only the
    # explicit True marker set by a successful registration counts.
    ("    _EXL3_OPS_READY = bool(getattr(torch.ops, \"_exl3_ops_ready\", False))\n",
     "    _EXL3_OPS_READY = getattr(torch.ops, \"_exl3_ops_ready\", False) is True\n"),
    # torch 2.14 regression fix: torch.ops attribute access resolves unknown
    # names to child _OpNamespace objects instead of raising, so the image's
    # hasattr guard never installs the dicts and item assignment dies with
    # TypeError: '_OpNamespace' object does not support item assignment.
    # Stash the registries in the torch.ops instance dict directly; setdefault
    # keeps them shared across module re-imports (the stash's original intent).
    ("    if not hasattr(torch.ops, \"_exl3_opaque_layers\"):\n"
     "        setattr(torch.ops, \"_exl3_opaque_layers\", {})\n"
     "    if not hasattr(torch.ops, \"_exl3_opaque_methods\"):\n"
     "        setattr(torch.ops, \"_exl3_opaque_methods\", {})\n"
     "    _EXL3_OPAQUE_LAYERS: dict[str, Any] = getattr(torch.ops, \"_exl3_opaque_layers\")\n"
     "    _EXL3_OPAQUE_METHODS: dict[str, Any] = getattr(torch.ops, \"_exl3_opaque_methods\")\n",
     "    _EXL3_OPAQUE_LAYERS: dict[str, Any] = torch.ops.__dict__.setdefault(\n"
     "        \"_exl3_opaque_layers\", {}\n"
     "    )\n"
     "    _EXL3_OPAQUE_METHODS: dict[str, Any] = torch.ops.__dict__.setdefault(\n"
     "        \"_exl3_opaque_methods\", {}\n"
     "    )\n"),
    # 89e442d: strip the VL wrapper prefix in _bf16_shards_for.
    ("    def _bf16_shards_for(self, prefix: str) -> list[int]:\n"
     "        \"\"\"Get bf16 shard indices for a non_routed_exl3 layer from the layers dict.\"\"\"\n"
     "        if not self.non_routed_exl3:\n"
     "            return []\n"
     "        layers = self.non_routed_exl3.get(\"layers\", {})\n",
     "    def _bf16_shards_for(self, prefix: str) -> list[int]:\n"
     "        \"\"\"Get bf16 shard indices for a non_routed_exl3 layer from the layers dict.\"\"\"\n"
     "        if not self.non_routed_exl3:\n"
     "            return []\n"
     "        # create_weights queries with the raw layer.prefix, which VL wrappers\n"
     "        # (glm5next) construct wrapper-prefixed (language_model.model.layers.N...)\n"
     "        # while coverage keys are inner-relative (model.layers.N...), exactly\n"
     "        # like the names the wrapper's load path yields after stripping.\n"
     "        # get_quant_method already normalizes this prefix away; strip it here\n"
     "        # too so the exact layers-dict lookup agrees with the routing decision.\n"
     "        if prefix.startswith(\"language_model.\"):\n"
     "            prefix = prefix[len(\"language_model.\"):]\n"
     "        layers = self.non_routed_exl3.get(\"layers\", {})\n"),
    # 682b3ca: pad only the EXL3 shards; bf16_shards rows keep true sizes.
    ("        in_per_partition = _exl3_pad128(true_in)\n"
     "        output_partition_sizes = [_exl3_pad128(s) for s in true_out_sizes]\n",
     "        in_per_partition = _exl3_pad128(true_in)\n"
     "        # Pad only the EXL3 shards: bf16_shards rows hold plain BF16\n"
     "        # weights and need no 128 alignment (GLM-5.3 KDA in_proj b/f_a/g_a\n"
     "        # shards are 64/128/128 rows).\n"
     "        output_partition_sizes = [\n"
     "            _exl3_pad128(s) if i not in bf16_shards else s\n"
     "            for i, s in enumerate(true_out_sizes)\n"
     "        ]\n"),
    ("if nr_bits is not None and nr_bits not in (2, 3, 4, 5, 6):",
     "if nr_bits is not None and nr_bits not in (2, 3, 4, 5, 6, 7, 8):"),
    ("            if k not in (2, 3, 4, 5, 6):",
     "            if k not in (2, 3, 4, 5, 6, 7, 8):"),
    ("if layer_bits is not None and layer_bits not in (2, 3, 4, 5, 6):",
     "if layer_bits is not None and layer_bits not in (2, 3, 4, 5, 6, 7, 8):"),
):
    print("exl3.py:", patch(E, old, new))
import ast
for f in (f"{V}/kda.py", f"{V}/model.py", E):
    ast.parse(open(f).read())
print("GV_BOOT_PATCH_OK")


# --- self-test: the lookups this overlay depends on must resolve ---
import json
MODEL = "/models/GLM-5.3-Flash-SAGE-REXL3-2.48bpw-vllm-overlay"
q = json.load(open(f"{MODEL}/config.json"))["quantization_config"]
from vllm_exl3.exl3 import Exl3Config
cfg = Exl3Config.from_config(q)
WRAP = "language_model.model.layers.{L}.self_attn.in_proj_qkvbfg_a"
INNER = "model.layers.{L}.self_attn.in_proj_qkvbfg_a"
for L, want_bits in ((0, 8), (4, 6)):
    got = cfg._bf16_shards_for(WRAP.format(L=L))
    assert got == [3, 4, 5], f"layer {L} wrapper-prefixed bf16_shards: {got}"
    got_inner = cfg._bf16_shards_for(INNER.format(L=L))
    assert got_inner == [3, 4, 5], f"layer {L} inner bf16_shards: {got_inner}"
    bits = cfg._bits_for_non_routed(INNER.format(L=L))
    assert bits == want_bits, f"layer {L} bits: {bits} != {want_bits}"
o8 = cfg._bits_for_non_routed("model.layers.0.self_attn.o_proj")
o7 = cfg._bits_for_non_routed("model.layers.3.self_attn.o_proj")
assert o8 == 8 and o7 == 7, f"o_proj bits: {o8}, {o7}"
assert cfg._matches_non_routed_exl3(INNER.format(L=0))
import vllm_exl3.exl3 as _exm
assert isinstance(_exm._EXL3_OPAQUE_LAYERS, dict), type(_exm._EXL3_OPAQUE_LAYERS)
assert isinstance(_exm._EXL3_OPAQUE_METHODS, dict), type(_exm._EXL3_OPAQUE_METHODS)
probe = type("P", (), {"prefix": "gv.selftest.linear"})()
nm = _exm._exl3_register_opaque_layer(probe, "linear", None)
assert _exm._EXL3_OPAQUE_LAYERS[nm] is probe
import torch
_op = getattr(torch.ops.vllm, "exl3_linear_forward", "MISS")
assert type(_op).__name__ not in ("str", "NoneType"), f"exl3_linear_forward not registered: {type(_op)}"
print("GV_BF16_LOOKUP_SELFTEST_OK")
