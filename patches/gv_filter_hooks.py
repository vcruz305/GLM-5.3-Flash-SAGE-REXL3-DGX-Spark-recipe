"""LANE GV loader filter for the m31-flash vllm snapshot (v3: source-scoped).

vLLM streams every tensor of every shard file, not just the index's
weight_map. The loaded index does not match the raw shard contents, so the
stream must be filtered to the pack's model.safetensors.index.json.

v2 bug (root cause of DFlash2 zero acceptance): the filter was applied to
EVERY source in the process. The DFlash2 drafter (/drafter/model.safetensors)
is loaded in the same process and none of its tensors are in the overlay
index, so every drafter tensor (fc.weight, candidate_selector.*, conv
kernels, qkv/o/mlp) was silently dropped and the drafter kept its zero
initialization -> constant draft token, 0% acceptance.

v3: filter ONLY the source whose model_or_path is the overlay dir
(VLLM_EXL3_MODEL_DIR); pass every other source through untouched. Count
dropped vs passed tensors per source and log the totals.
"""
import json
import importlib.abc
import importlib.util
import os
import sys

_DIR = os.environ.get("VLLM_EXL3_MODEL_DIR", "")
try:
    _TARGET_RP = os.path.realpath(_DIR) if _DIR else ""
except Exception:
    _TARGET_RP = ""
try:
    with open(os.path.join(_DIR, "model.safetensors.index.json")) as _f:
        _KEEP = frozenset(json.load(_f)["weight_map"])
except Exception:
    _KEEP = None
    print(f"GV_FILTER: WARNING no overlay index in {_DIR!r}; no filtering", flush=True)

_TARGETS = {"vllm.model_executor.model_loader.default_loader": "_patch"}


def _wrap(orig):
    def filtered(self, source, *args, **kwargs):
        src = getattr(source, "model_or_path", "") or ""
        try:
            rp = os.path.realpath(src)
        except Exception:
            rp = src
        is_target = bool(_TARGET_RP) and rp == _TARGET_RP
        dropped = 0
        passed = 0
        it = orig(self, source, *args, **kwargs)
        if not is_target or _KEEP is None:
            for n, t in it:
                passed += 1
                yield n, t
            print(f"GV_FILTER PASS-THROUGH src={src} passed={passed}", flush=True)
            return
        for n, t in it:
            if n in _KEEP:
                passed += 1
                yield n, t
            else:
                dropped += 1
        print(
            f"GV_FILTER OVERLAY src={src} passed={passed} dropped={dropped}",
            flush=True,
        )

    return filtered


def _wrap_all(orig):
    def counted(self, model_config, model):
        passed = 0
        for n, t in orig(self, model_config, model):
            passed += 1
            yield n, t
        print(f"GV_FILTER get_all_weights passed={passed}", flush=True)

    return counted


def _patch(mod):
    cls = mod.DefaultModelLoader
    if getattr(cls, "_gv_filter_v3", False):
        print("GV_FILTER: already patched (v3)", flush=True)
        return
    it = getattr(cls, "_get_weights_iterator", None)
    if it is not None:
        setattr(cls, "_get_weights_iterator", _wrap(it))
    allw = getattr(cls, "get_all_weights", None)
    if allw is not None:
        setattr(cls, "get_all_weights", _wrap_all(allw))
    cls._gv_filter_v3 = True
    print("GV_FILTER v3 source-scoped filter installed", flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in _TARGETS:
            return None
        fn = globals()[_TARGETS.pop(name)]
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            if _TARGETS:
                sys.meta_path.insert(0, self)
        loader = spec.loader
        exec_module = loader.exec_module

        def _exec(module):
            exec_module(module)
            fn(module)

        loader.exec_module = _exec
        return spec


if not getattr(sys, "_gv_filter_hooks", False):
    sys._gv_filter_hooks = True
    sys.meta_path.insert(0, _Finder())
