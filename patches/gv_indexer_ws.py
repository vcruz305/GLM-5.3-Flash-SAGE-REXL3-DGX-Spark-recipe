#!/usr/bin/env python3
"""gv_indexer_ws.py -- sparse-indexer prefill workspace right-size, re-derived
from the reference GLM53_INDEXER_WORKSPACE=rightsize mechanism
(glm-5.3-one-spark docs/DESIGN-indexer-workspace.md) against THIS snapshot.

Our snapshot already divides by compress_ratio at the metadata-builder call
site (indexer.py:1016-1019), so at MAXLEN 262144 the process-locked workspace
is 262144x40/4 = 2,621,440 entries x 132 B = 330 MiB. No legal step can ever
present more than min(max_num_seqs, max_num_batched_tokens) requests x
cdiv(max_model_len + num_speculative_tokens, compress_ratio) compressed rows
(one whole max-length request is the hard floor and is covered); at this
lane's config (MAXSEQS=1, K5, kpool 4) that is 65,538 entries = 8.65 MB.
The freed ~320 MiB is what the 262K spec boot needs to clear the 0.5 GiB
MemAvailable guard floor.

Safety (the reference's law, kept):
- CLAMPED AT STOCK: rightsize = min(stock, legal) — the patch can only ever
  narrow; a config whose legal max exceeds stock returns stock unchanged.
- FAIL-CLOSED CROSS-CHECK: hf_text_config.index_kpool must equal the
  resolved compress_ratio, else the boot dies naming both values.
- The workspace manager's _ensure_workspace_size AssertionError stays as the
  runtime backstop; chunk lists are provably identical for every legal batch
  (the reference's test-receipt), so the backstop never fires in operation.
- Env-gated: GV_INDEXER_WS unset/empty = stock, byte-identical.

Mode: GV_INDEXER_WS=1 activates; every activation logs the stock, legal, and
rightsize entry counts as the boot receipt.
"""

from __future__ import annotations

import ast
import os
import sys

IDX = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"

MARK = "GV INDEXER-WS"

EDIT_OLD = '''\
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        # Counted in compressed rows, like the chunker's seq_lens and the
        # workspace.
        self.max_prefill_buffer_size = (
            get_max_prefill_buffer_size(self.vllm_config) // self.compress_ratio
        )
'''

EDIT_NEW = '''\
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        # Counted in compressed rows, like the chunker's seq_lens and the
        # workspace.
        _gv_stock_entries = (
            get_max_prefill_buffer_size(self.vllm_config) // self.compress_ratio
        )
        self.max_prefill_buffer_size = _gv_stock_entries
        # {MARK}: right-size the prefill workspace to the legal maximum a
        # single step can present (the reference's rightsize formula,
        # re-derived against this snapshot). CLAMPED AT STOCK: the patch can
        # only ever narrow. The workspace manager's _ensure_workspace_size
        # AssertionError remains the runtime backstop.
        import os as _os
        if _os.environ.get("GV_INDEXER_WS"):
            _gv_spec = self.vllm_config.speculative_config
            _gv_k = (
                _gv_spec.num_speculative_tokens
                if _gv_spec is not None else 0
            )
            _gv_reqs = max(1, min(
                self.vllm_config.scheduler_config.max_num_seqs,
                self.vllm_config.scheduler_config.max_num_batched_tokens,
            ))
            _gv_ratio = max(1, self.compress_ratio)
            _gv_legal = _gv_reqs * -(-(
                self.vllm_config.model_config.max_model_len + _gv_k
            ) // _gv_ratio)
            _gv_kpool = getattr(
                self.vllm_config.model_config.hf_text_config,
                "index_kpool", None,
            )
            assert _gv_kpool == self.compress_ratio, (
                "{MARK}: hf_text_config.index_kpool "
                f"{{_gv_kpool}} != compress_ratio {{self.compress_ratio}}; "
                "refusing to serve"
            )
            self.max_prefill_buffer_size = min(_gv_stock_entries, _gv_legal)
            print(
                f"[{MARK}] stock_entries={{_gv_stock_entries}} "
                f"legal_entries={{_gv_legal}} "
                f"rightsize_entries={{self.max_prefill_buffer_size}}"
            )
'''.format(MARK=MARK)


def static_selftests() -> None:
    # T1: clamped at stock — min() is the only assignment to the field.
    assert EDIT_NEW.count("self.max_prefill_buffer_size = ") == 2
    assert "min(_gv_stock_entries, _gv_legal)" in EDIT_NEW
    # T2: the fail-closed cross-check is unconditional when active.
    assert "assert _gv_kpool == self.compress_ratio" in EDIT_NEW
    # T3: the hard floor — the request count is at least 1 and the span
    # includes max_model_len (cdiv is the safe side).
    assert "max(1, min(" in EDIT_NEW
    assert "max_model_len + _gv_k" in EDIT_NEW
    # T4: env-gated stock default.
    assert 'if _os.environ.get("GV_INDEXER_WS"):' in EDIT_NEW
    # T5: the stock expression survives verbatim under its renamed local.
    assert (
        "get_max_prefill_buffer_size(self.vllm_config) // self.compress_ratio"
        in EDIT_NEW
    )
    print(f"[{MARK}] static selftests: PASS")


def main() -> int:
    mode = os.environ.get("GV_INDEXER_WS", "")
    if not mode:
        print(f"[{MARK}] GV_INDEXER_WS unset, inactive")
        return 0
    static_selftests()
    text = open(IDX, encoding="utf-8").read()
    if MARK in text:
        print(f"[{MARK}] {IDX}: already patched; no-op")
        return 0
    n = text.count(EDIT_OLD)
    assert n == 1, (
        f"{IDX}: anchor found {n} times, expected 1 -- upstream drift; "
        "re-derive before building"
    )
    text = text.replace(EDIT_OLD, EDIT_NEW, 1)
    ast.parse(text, filename=IDX)
    open(IDX, "w", encoding="utf-8").write(text)
    print(f"[{MARK}] rightsize clamp: applied")
    print(f"[{MARK}] BOOT PATCH OK (mode={mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
