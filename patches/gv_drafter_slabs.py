#!/usr/bin/env python3
"""gv_drafter_slabs.py -- DFlash2 drafter disjoint-slab KV partition, re-derived
against THIS snapshot's generic planner (v1/core/kv_cache_planning.py), plus
the hybrid prefix-hit companion (v1/core/kv_cache_coordinator.py).

The reference (K2 recipe overlay/patch_glm5_drafter_group.py) targets a
_get_kv_cache_groups_glm5_next fast path that does NOT exist in this tree; the
mechanism is re-derived here against the generic planner.

MECHANISM, re-derived (see GLM_FLASH_VLLM_LOG '## LANE PORT'):
- At planning time the drafter's KV spec arrives page-unified with the target
  (its block has been bumped to the global attention block, its per-token
  page is the largest in the model), so the shared pool's bytes-per-block and
  the drafter's charged page count both scale with max_model_len (the
  measured ~20.5 KB/token linear term).
- PORT: partition the drafter's is_draft layers out of every unified-page
  path, run the stock planner on the rest (byte-identical target layout,
  per-block pool bytes unchanged), re-issue the drafter as ONE uniform group
  of exact SlidingWindowSpec at its real kernel block (256: the size the
  stock worker already splits the page-unified manager block into), and
  append that group LAST so existing group ids stay stable. The
  SlidingWindow demand law then bounds the drafter's charge to its window +
  in-flight blocks instead of context-proportional pages, and the runtime
  SlidingWindowManager frees out-of-window blocks exactly as the drafter's
  attention reads.
- NEVER emit page_size_padded for the drafter: manager block == kernel block
  means the worker's select_common_block_size returns it unsplit (the
  reference's boot-8 OOB lesson), and the re-issue forces padded=None.
- COMPANION (required with the port): in the hybrid coordinator's prefix-hit
  fixed point the drafter's window-bounded group must not shrink the
  reconciled hit length; otherwise every long-prefix re-hit re-prefills the
  whole target (TTFT explosion). The drafter keeps its own window-bounded
  blocks/length so it re-prefills only its window. The EAGLE group
  annotation (planning.py) already flags the drafter group in this snapshot,
  so the target groups are not eagle-dropped.

Modes (env GV_DRAFTER_SLABS): unset/0 = inactive; 1 = port + instrumentation.
Instrumentation: full spec/group/pool/demand decomposition (patched AND a
shadow stock-planner run for comparison) is appended to
/mem/drafter_kv_dump.json; on any port assertion failure the same dump is
written before re-raising, so a failed boot still yields the ground truth.
"""

from __future__ import annotations

import ast
import os
import sys

PLAN = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_planning.py"
COORD = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py"

MARK = "GV DRAFTER-SLABS"

# ---------------------------------------------------------------------------
# Edit A (module level, at the stock def anchor): rename the stock grouping
# entry point and install the dump hook, the fail-dump, the partition wrapper
# and the guard. The stock body below the anchor is untouched.
# ---------------------------------------------------------------------------

EDIT_A_OLD = '''\
def get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
'''

EDIT_A_NEW = '''\
def _gv_drafter_slabs_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """{MARK}: partition the spec-decode drafter's is_draft layers out of every
    unified-page path, run the stock planner on the rest, append the re-issued
    drafter group LAST (stable group ids). Fail closed on every invariant."""
    import dataclasses as _dc

    draft_names = sorted(
        name for name, spec in kv_cache_spec.items()
        if getattr(spec, "is_draft", False)
    )
    if not draft_names:
        return _gv_stock_get_kv_cache_groups(vllm_config, kv_cache_spec)
    draft_specs = {{name: kv_cache_spec[name] for name in draft_names}}
    rest = {{n: s for n, s in kv_cache_spec.items() if n not in draft_specs}}
    first = draft_specs[draft_names[0]]
    assert all(draft_specs[n] == first for n in draft_names), (
        "{MARK}: drafter layers do not share one spec"
    )
    window = getattr(first, "sliding_window", None)
    assert isinstance(window, int) and window > 0, (
        "{MARK}: drafter spec has no sliding window; refusing to partition"
    )
    # Manager block == the SWA kernel block the stock worker already splits
    # the page-unified manager block into (256 on this platform): the
    # worker's select_common_block_size returns it unsplit, so no
    # page_size_padded is ever emitted and no per-kernel-block stride can
    # leave the page (the reference's boot-8 OOB lesson). A backend that
    # does not accept it fails closed at worker init.
    compact = 256
    assert compact <= first.block_size, (
        "{MARK}: compact block exceeds the manager block"
    )
    fields = {{f.name: getattr(first, f.name) for f in _dc.fields(first)
              if f.init}}
    allowed = {{f.name for f in _dc.fields(SlidingWindowSpec)}}
    new_specs = {{}}
    for name in draft_names:
        base = dict(fields)
        base["block_size"] = compact
        base["sliding_window"] = window
        # NEVER emit a padded page for the drafter (boot-8 lesson), and never
        # inherit the page-unified layout's block stride: the slab is disjoint
        # and its manager block is the kernel block (unsplit).
        base["page_size_padded"] = None
        base["block_stride_alignment"] = None
        base["is_draft"] = True
        base["extra_retained_tokens"] = getattr(
            first, "extra_retained_tokens", 0
        )
        new_specs[name] = SlidingWindowSpec(
            **{{k: v for k, v in base.items() if k in allowed}}
        )
    draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_specs)
    assert draft_uniform is not None, (
        "{MARK}: re-issued drafter specs are not uniform-type"
    )
    groups = _gv_stock_get_kv_cache_groups(vllm_config, rest)
    assert groups, "{MARK}: stock planner returned no groups"
    # Append LAST: every pre-existing group id is unchanged.
    return groups + [KVCacheGroupSpec(list(new_specs), draft_uniform)]


def _gv_drafter_slabs_fail(vllm_config, kv_cache_spec, err) -> None:
    """Ground-truth dump on a port failure, then the caller re-raises."""
    try:
        import json as _json
        stock = _gv_stock_get_kv_cache_groups(vllm_config, dict(kv_cache_spec))
        rows = []
        for gid, g in enumerate(stock):
            rows.append({{
                "gid": gid,
                "layers": len(g.layer_names),
                "cls": type(g.kv_cache_spec).__name__,
                "block": getattr(g.kv_cache_spec, "block_size", None),
                "page": getattr(g.kv_cache_spec, "page_size_bytes", None),
            }})
        with open("/mem/drafter_kv_dump.json", "a") as f:
            f.write(_json.dumps({{
                "mode": "fail", "error": repr(err),
                "stock_groups": rows,
            }}) + "\\n")
    except Exception as e:  # noqa: BLE001 - never mask the original failure
        print("{MARK}: fail-dump error", repr(e))


def get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    import os as _os
    if not _os.environ.get("GV_DRAFTER_SLABS"):
        return _gv_stock_get_kv_cache_groups(vllm_config, kv_cache_spec)
    try:
        return _gv_drafter_slabs_groups(vllm_config, kv_cache_spec)
    except AssertionError as e:
        _gv_drafter_slabs_fail(vllm_config, kv_cache_spec, e)
        raise


def _gv_drafter_slabs_dump(vllm_config, merged, groups):
    """{MARK}: full spec/group/pool/demand decomposition (patched + shadow
    stock) for the port's acceptance test. Pure probes only. Fail-closed on
    the port invariants (A1-A5)."""
    import json as _json
    import os as _os

    def _inner(s):
        specs = getattr(s, "kv_cache_specs", None)
        if isinstance(specs, dict) and specs:
            return next(iter(specs.values()))
        return s

    def _spec_row(s):
        i = _inner(s)
        return {{
            "cls": type(i).__name__,
            "block": getattr(i, "block_size", None),
            "page": getattr(i, "page_size_bytes", None),
            "padded": getattr(i, "page_size_padded", None),
            "window": getattr(i, "sliding_window", None),
            "extra_retained": getattr(i, "extra_retained_tokens", None),
            "is_draft": bool(getattr(i, "is_draft", False)),
        }}

    def _group_pages(g):
        s = g.kv_cache_spec
        try:
            return s.max_memory_usage_pages(vllm_config)
        except Exception:
            from vllm.utils.math_utils import cdiv as _c
            return _c(s.max_memory_usage_bytes(vllm_config),
                     s.page_size_bytes)

    def _decomp(gs):
        rows = []
        total_pages = 0
        for gid, g in enumerate(gs):
            pages = _group_pages(g)
            rows.append({{
                "gid": gid,
                "layers": len(g.layer_names),
                "group_block": getattr(g.kv_cache_spec, "block_size", None),
                "group_page_sum": getattr(
                    g.kv_cache_spec, "page_size_bytes", None
                ),
                "pages": pages,
                "eagle": bool(getattr(g, "is_eagle_group", False)),
                "spec": _spec_row(g.kv_cache_spec),
            }})
            total_pages += pages
        ppb = _pool_bytes_per_block(vllm_config, gs) if gs else 0
        return {{"groups": rows, "pool_bytes_per_block": ppb,
                "total_pages": total_pages,
                "demand_bytes": ppb * total_pages}}

    out = {{
        "mode": _os.environ.get("GV_DRAFTER_SLABS", ""),
        "max_model_len": vllm_config.model_config.max_model_len,
        "mnbt": vllm_config.scheduler_config.max_num_batched_tokens,
        "max_in_flight_tokens": vllm_config.max_in_flight_tokens,
        "num_speculative_tokens": (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None else None
        ),
        "patched": _decomp(groups),
    }}
    try:
        stock_groups = _gv_stock_get_kv_cache_groups(vllm_config, dict(merged))
        out["shadow_stock"] = _decomp(stock_groups)
    except Exception as e:  # noqa: BLE001
        out["shadow_stock"] = {{"error": repr(e)}}
    if out["mode"] == "1":
        draft_rows = [r for r in out["patched"]["groups"]
                     if r["spec"]["cls"] == "SlidingWindowSpec"
                     and r["spec"]["is_draft"]]
        ppb_all = out["patched"]["pool_bytes_per_block"]
        rest_groups = [g for g in groups
                       if type(_inner(g.kv_cache_spec)).__name__
                       != "SlidingWindowSpec"]
        # A5: exactly one re-issued drafter group, appended LAST.
        assert len(draft_rows) == 1 and draft_rows[0]["gid"] == len(groups) - 1, (
            "{MARK} A5 FAILED: drafter group is not the single LAST group",
            [r["gid"] for r in draft_rows],
        )
        # A4: no padded page on any drafter spec.
        assert all(r["spec"]["padded"] is None for r in draft_rows), (
            "{MARK} A4 FAILED: drafter spec is padded"
        )
        # A2: the drafter's disjoint slab is strictly smaller than the pool's
        # per-block bytes (its slab rides inside the block, adds no bytes).
        slab = sum(r["group_page_sum"] for r in draft_rows)
        assert 0 < slab < ppb_all, (
            "{MARK} A2 FAILED: drafter slab not inside the pool block",
            slab, ppb_all,
        )
        out["drafter_slab_bytes_per_block"] = slab
        # A1: per-block pool bytes UNCHANGED by the drafter group (KV
        # capacity preserved for the target).
        ppb_rest = _pool_bytes_per_block(vllm_config, rest_groups) \
            if rest_groups else 0
        assert ppb_all == ppb_rest, (
            "{MARK} A1 FAILED: pool per-block bytes changed",
            ppb_all, ppb_rest,
        )
        # A3: the drafter's charged pages are exactly its window-bounded
        # admission demand (window + in-flight, never context-proportional).
        d = draft_rows[0]
        win = d["spec"]["window"]
        extra = d["spec"]["extra_retained"] or 0
        infl = vllm_config.max_in_flight_tokens
        mml = vllm_config.model_config.max_model_len
        from vllm.utils.math_utils import cdiv as _c
        expect = _c(min(win - 1 + extra + infl, mml),
                   d["spec"]["block"]) + 1
        assert d["pages"] == expect, (
            "{MARK} A3 FAILED: drafter pages are not window-bounded",
            d["pages"], expect,
        )
        out["drafter_window_pages"] = expect
    with open("/mem/drafter_kv_dump.json", "a") as f:
        f.write(_json.dumps(out) + "\\n")


def _gv_stock_get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
'''.format(MARK=MARK)

# ---------------------------------------------------------------------------
# Edit B: call the dump hook from get_kv_cache_configs once the (possibly
# ported) groups exist. Only inside the speculative branch, so AR boots are
# untouched.
# ---------------------------------------------------------------------------

EDIT_B_OLD = '''\
        for group in global_kv_cache_groups:
            group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)
        _warn_if_unannotated_eagle_mamba(spec_config, global_kv_cache_groups)
'''

EDIT_B_NEW = '''\
        for group in global_kv_cache_groups:
            group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)
        _warn_if_unannotated_eagle_mamba(spec_config, global_kv_cache_groups)
        _gv_drafter_slabs_dump(
            vllm_config, merged_kv_cache_specs, global_kv_cache_groups
        )
'''

# ---------------------------------------------------------------------------
# Edit C: the hybrid prefix-hit companion. The drafter's window-bounded group
# must not shrink the reconciled hit length (the reference's
# patch_hybrid_prefix_hit.py mechanism, re-anchored byte-exact against this
# snapshot's coordinator).
# ---------------------------------------------------------------------------

EDIT_C_OLD = '''\
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)
'''

EDIT_C_NEW = '''\
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                if (  # {MARK}: the spec-decode drafter's exact-SlidingWindow
                    # group is window-bounded and must not shrink the
                    # reconciled hybrid hit (else every long-prefix re-hit
                    # re-prefills the whole target). Keep its own blocks and
                    # window-bounded length so the drafter re-prefills only
                    # its window; leave them empty when the cached window does
                    # not cover the hit so a fresh window is allocated.
                    type(next(iter(getattr(
                        spec, "kv_cache_specs", {{None: None}}
                    ).values()))).__name__ == "SlidingWindowSpec"
                ):
                    if _new_hit_length >= curr_hit_length:
                        for group_id, blocks in zip(group_ids, hit_blocks):
                            hit_blocks_by_group[group_id] = blocks
                            hit_length_by_group[group_id] = _new_hit_length
                    continue
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)
'''.format(MARK=MARK)


def apply(path: str, old: str, new: str, label: str) -> None:
    text = open(path, encoding="utf-8").read()
    n = text.count(old)
    assert n == 1, (
        f"{path}: anchor for [{label}] found {n} times, expected 1 -- "
        "the upstream file has drifted; re-derive before building"
    )
    text = text.replace(old, new, 1)
    ast.parse(text, filename=path)
    open(path, "w", encoding="utf-8").write(text)
    print(f"[{MARK}] {label}: applied")


def static_selftests() -> None:
    """Fail-closed static checks on the emitted port source."""
    # T3: the drafter re-issue never emits a padded page: the only place the
    # re-issue touches page_size_padded is clearing it to None.
    assert EDIT_A_NEW.count('base["page_size_padded"]') == 1
    assert 'base["page_size_padded"] = None' in EDIT_A_NEW
    # T4: the drafter group is appended LAST (stable group ids).
    assert "return groups + [KVCacheGroupSpec(" in EDIT_A_NEW
    # T6: the partition predicate is is_draft (robust to either spec class
    # the drafter arrives as), and the re-issue is exact SlidingWindowSpec.
    assert 'getattr(spec, "is_draft", False)' in EDIT_A_NEW
    assert "SlidingWindowSpec(\n" in EDIT_A_NEW
    # T5: the companion excludes ONLY the drafter's exact SlidingWindowSpec
    # group (the target's MLA groups and the tail fall through untouched).
    assert EDIT_C_NEW.count('"SlidingWindowSpec"') == 1
    assert EDIT_C_NEW.count("curr_hit_length = _new_hit_length") == 1
    # T7: all A1-A5 asserts are present and fail-closed.
    for a in ("A1 FAILED", "A2 FAILED", "A3 FAILED", "A4 FAILED", "A5 FAILED"):
        assert a in EDIT_A_NEW
    # T8: the stock entry point survives verbatim under its renamed def.
    assert EDIT_A_NEW.endswith(
        "def _gv_stock_get_kv_cache_groups(\n"
        "    vllm_config: VllmConfig,\n"
        "    kv_cache_spec: dict[str, KVCacheSpec],\n"
        ") -> list[KVCacheGroupSpec]:\n"
    )
    # T9: pure probes only in the dump (no builder state mutation).
    assert "from_builder" not in EDIT_A_NEW
    print(f"[{MARK}] static selftests: PASS")


def main() -> int:
    mode = os.environ.get("GV_DRAFTER_SLABS", "")
    if not mode:
        print(f"[{MARK}] GV_DRAFTER_SLABS unset, inactive")
        return 0
    static_selftests()
    plan = open(PLAN, encoding="utf-8").read()
    if MARK in plan:
        print(f"[{MARK}] {PLAN}: already patched; no-op")
    else:
        for required in (
            "def get_kv_cache_groups(",
            "_max_memory_usage_bytes_from_groups",
            "_warn_if_unannotated_eagle_mamba",
            "SlidingWindowSpec",
            "UniformTypeKVCacheSpecs",
            "_pool_bytes_per_block(",
        ):
            assert required in plan, (
                f"ANCHOR PRECHECK FAILED: {required!r} not in {PLAN}"
            )
        apply(PLAN, EDIT_A_OLD, EDIT_A_NEW, "planning: partition wrapper")
        apply(PLAN, EDIT_B_OLD, EDIT_B_NEW, "planning: dump hook call")
    coord = open(COORD, encoding="utf-8").read()
    if MARK in coord:
        print(f"[{MARK}] {COORD}: already patched; no-op")
    else:
        apply(COORD, EDIT_C_OLD, EDIT_C_NEW, "companion: hybrid hit min")
    print(f"[{MARK}] BOOT PATCH OK (mode={mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
