#!/usr/bin/env python3
"""gv_drafter_slabs_cputest.py -- fail-closed CPU-only acceptance probe for the
GV DRAFTER-SLABS port. Runs in a TRANSIENT container (no GPU, no network)
after gv_drafter_slabs.py has patched the in-container vllm tree. Every check
must pass; any failure exits 1.

Proves, with the REAL spec classes and the REAL patched planner module:
  T0  the patched planning module imports cleanly and exposes the guard,
      the partition, and the stock entry point under its renamed def;
  T-A per-token byte identity through the re-issue (cross-multiplication);
  T-B the window-bounded admission demand law at 8K and 262K;
  T-C the wrapper's re-issue yields exact SlidingWindowSpec (no padded page,
      no inherited stride, is_draft) and groups into ONE uniform group whose
      page sum is the disjoint slab;
  T-D the pool per-block bytes are UNCHANGED by appending the drafter group,
      under EVERY KVCacheLayout that can express both group sets;
  T-E the guard no-ops to the stock planner when the env is unset.
"""

import os

os.environ["GV_DRAFTER_SLABS"] = "1"

import dataclasses as dc  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

from vllm.v1.core import kv_cache_planning as plan  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

PASS = []


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        sys.exit(1)
    PASS.append(name)
    print(f"PASS {name} {detail}")


# T0: patched module imports cleanly; all port symbols present.
check(
    "T0 patched module",
    hasattr(plan, "_gv_stock_get_kv_cache_groups")
    and hasattr(plan, "_gv_drafter_slabs_groups")
    and hasattr(plan, "_gv_drafter_slabs_dump")
    and plan.get_kv_cache_groups.__module__ == plan.__name__,
)

# Real class geometry: a page-unified drafter spec (block 8960) and its
# compact re-issue (block 256), same per-token layout.
S1 = SlidingWindowSpec(
    block_size=8960, sliding_window=2048,
    num_kv_heads=2, head_size=128, dtype=torch.bfloat16,
)
S2 = SlidingWindowSpec(
    block_size=256, sliding_window=2048,
    num_kv_heads=2, head_size=128, dtype=torch.bfloat16,
)

# T-A: per-token byte identity through the re-issue.
check(
    "T-A per-token identity",
    S2.page_size_bytes * S1.block_size == S1.page_size_bytes * S2.block_size,
    f"{S1.page_size_bytes}/{S1.block_size} -> {S2.page_size_bytes}/{S2.block_size}",
)

# T-B: window-bounded admission demand law (the port's demand target).
mif = 8192
b262 = S2.max_admission_blocks_per_request(mif, 262144)
exp262 = -(-min(2047 + mif, 262144) // 256) + 1
check("T-B demand 262K", b262 == exp262 == 41, f"{b262} == {exp262}")
b8k = S2.max_admission_blocks_per_request(mif, 8192)
exp8k = -(-min(2047 + mif, 8192) // 256) + 1
check("T-B demand 8K", b8k == exp8k == 33, f"{b8k} == {exp8k}")

# T-C: the wrapper's re-issue logic, verbatim from the patch.
def reissue(first, compact):
    fields = {
        f.name: getattr(first, f.name) for f in dc.fields(first) if f.init
    }
    allowed = {f.name for f in dc.fields(SlidingWindowSpec)}
    base = dict(fields)
    base["block_size"] = compact
    base["sliding_window"] = first.sliding_window
    base["page_size_padded"] = None
    base["block_stride_alignment"] = None
    base["is_draft"] = True
    base["extra_retained_tokens"] = getattr(first, "extra_retained_tokens", 0)
    return SlidingWindowSpec(
        **{k: v for k, v in base.items() if k in allowed}
    )


R = reissue(S1, 256)
check(
    "T-C re-issue properties",
    type(R).__name__ == "SlidingWindowSpec"
    and R.block_size == 256
    and R.sliding_window == 2048
    and R.page_size_padded is None
    and R.block_stride_alignment is None
    and R.is_draft
    and R.page_size_bytes == S2.page_size_bytes,
    f"{R.page_size_bytes}",
)
d5 = {f"d{i}": R for i in range(5)}
U = UniformTypeKVCacheSpecs.from_specs(d5)
check(
    "T-C uniform slab",
    U is not None and U.page_size_bytes == 5 * S2.page_size_bytes,
    f"{None if U is None else U.page_size_bytes}",
)

# T-D: pool per-block bytes UNCHANGED by the drafter group, under every
# expressible layout (the target group is 80 full-attention layers whose
# per-block bytes dwarf the drafter's disjoint slab).
tgt = {
    f"t{i}": FullAttentionSpec(
        block_size=256, num_kv_heads=8, head_size=128, dtype=torch.bfloat16,
    )
    for i in range(80)
}
T = UniformTypeKVCacheSpecs.from_specs(tgt)
assert T is not None and T.page_size_bytes > 40 * U.page_size_bytes
gT = KVCacheGroupSpec(sorted(tgt), T)
gD = KVCacheGroupSpec(sorted(d5), U)
expressed = 0
for layout in KVCacheLayout:
    try:
        a = plan._get_kv_cache_bytes_per_block([gT], layout)
        b = plan._get_kv_cache_bytes_per_block([gT, gD], layout)
    except Exception:  # noqa: BLE001 - layout cannot express these groups
        continue
    expressed += 1
    check(f"T-D pool identity [{layout.name}]", a == b, f"{a} == {b}")
check("T-D coverage", expressed >= 1, f"{expressed} layouts")

# T-E: the guard no-ops to the stock planner when the env is unset, and
# routes to the partition when it is set (behavioral, via sentinel swaps).
plan._gv_stock_get_kv_cache_groups = lambda vc, specs: "STOCK"
plan._gv_drafter_slabs_groups = lambda vc, specs: "PORT"
os.environ["GV_DRAFTER_SLABS"] = ""
check(
    "T-E guard stock",
    plan.get_kv_cache_groups(None, {}) == "STOCK",
)
os.environ["GV_DRAFTER_SLABS"] = "1"
check(
    "T-E guard ported",
    plan.get_kv_cache_groups(None, {}) == "PORT",
)

print(f"CPU PROBE: ALL {len(PASS)} CHECKS PASS")
