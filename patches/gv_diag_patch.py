#!/usr/bin/env python3
"""GV3 diagnostic patch v2 (in-container, run before vllm serve).

Active only when GV_DIAG=1. Instruments the DFlash speculator so the first
real propose() calls report: per-tap aux feature sanity, the combined anchor
embedding, live drafter weight forensics (fc vs the raw checkpoint file,
gemm dispatch vs torch.nn.functional.linear), the proposed draft tokens, and
position-0 agreement of the previous draft against the anchor token the
target subsequently sampled.
"""
import os


def patch(path, old, new):
    s = open(path).read()
    if s.count(new) == 1:
        return "already patched"
    if s.count(old) != 1:
        raise SystemExit(f"{path}: anchor count {s.count(old)} != 1")
    open(path, "w").write(s.replace(old, new))
    return "patched"


if not os.environ.get("GV_DIAG"):
    print("gv_diag_patch: GV_DIAG not set, inactive")
    raise SystemExit(0)

S = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"

print("os:", patch(S, "import copy\nfrom typing import Any\n",
      "import os\nimport copy\nfrom typing import Any\n"))

DIAG_METHOD = (
    "    def _gv_diag(self, num_reqs, aux_hidden_states, last_sampled):\n"
    "        try:\n"
    "            import json as _json, struct as _struct\n"
    "            import torch.nn.functional as _F\n"
    "            n = int(num_reqs)\n"
    "            self._gv_calls = getattr(self, '_gv_calls', 0) + 1\n"
    "            if self._gv_calls > 3:\n"
    "                return\n"
    "            torch.cuda.synchronize()\n"
    "            msg = [f'GV_DIAG2 call={self._gv_calls} reqs={n}']\n"
    "            inner = self.model.model\n"
    "            def _wstat(name, mod):\n"
    "                try:\n"
    "                    w = getattr(mod, 'weight', None)\n"
    "                    if w is None:\n"
    "                        msg.append(f'{name}: no weight attr')\n"
    "                        return\n"
    "                    wf = w.detach().float()\n"
    "                    msg.append(f'{name}: shape={tuple(w.shape)} absmean={wf.abs().mean().item():.6f} finite={bool(torch.isfinite(wf).all())}')\n"
    "                except Exception as e:\n"
    "                    msg.append(f'{name}: err {e!r}')\n"
    "            _wstat('fc.weight', getattr(inner, 'fc', None))\n"
    "            _wstat('hidden_norm.weight', getattr(inner, 'hidden_norm', None))\n"
    "            _wstat('embed_tokens.weight', getattr(inner, 'embed_tokens', None))\n"
    "            _wstat('layers0.qkv_proj.weight', inner.layers[0].self_attn.qkv_proj)\n"
    "            zero_params = []\n"
    "            for pn, pp in inner.named_parameters():\n"
    "                with torch.no_grad():\n"
    "                    if float(pp.detach().float().abs().sum()) == 0.0:\n"
    "                        zero_params.append(pn)\n"
    "            msg.append(f'live zero params ({len(zero_params)}): ' + ','.join(zero_params[:40]))\n"
    "            sel = getattr(inner, 'candidate_selector', None)\n"
    "            _wstat('selector.hidden_projection.weight', getattr(sel, 'hidden_projection', None))\n"
    "            try:\n"
    "                p = '/drafter/model.safetensors'\n"
    "                with open(p, 'rb') as f:\n"
    "                    hn = _struct.unpack('<Q', f.read(8))[0]\n"
    "                    hdr = _json.loads(f.read(hn))\n"
    "                off = hdr['fc.weight']['data_offsets']\n"
    "                with open(p, 'rb') as f:\n"
    "                    f.seek(8 + hn + off[0])\n"
    "                    raw = f.read(off[1] - off[0])\n"
    "                raw_t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(4096, 20480).cuda()\n"
    "                lw = inner.fc.weight.detach()\n"
    "                msg.append(f'raw fc absmean={raw_t.float().abs().mean().item():.6f} live_eq_raw={bool(torch.equal(raw_t, lw))}')\n"
    "                xr = torch.randn(8, 20480, device='cuda', dtype=torch.bfloat16)\n"
    "                with torch.no_grad():\n"
    "                    out_fn = _F.linear(xr, lw)\n"
    "                    out_mod = inner.fc(xr)\n"
    "                msg.append(f'fc(randn): F.linear absmean={out_fn.float().abs().mean().item():.6f} module absmean={out_mod.float().abs().mean().item():.6f}')\n"
    "            except Exception as e:\n"
    "                msg.append(f'fc rawtest err {e!r}')\n"
    "            if aux_hidden_states:\n"
    "                for i, h in enumerate(aux_hidden_states):\n"
    "                    hf = h[:64].float()\n"
    "                    msg.append(f'tap{i} shape={tuple(h.shape)} finite={bool(torch.isfinite(hf).all())} absmean={hf.abs().mean().item():.4f}')\n"
    "                with torch.no_grad():\n"
    "                    anchor = self.model.combine_hidden_states(torch.cat([h[:64] for h in aux_hidden_states], dim=-1))\n"
    "                    anchor_fn = _F.linear(torch.cat([h[:64] for h in aux_hidden_states], dim=-1), inner.fc.weight.detach())\n"
    "                af = anchor.float()\n"
    "                msg.append(f'anchor_combined module absmean={af.abs().mean().item():.6f} F.linear absmean={anchor_fn.float().abs().mean().item():.6f}')\n"
    "            else:\n"
    "                msg.append('NO_AUX_HIDDEN_STATES')\n"
    "            prev = getattr(self, '_gv_prev_draft0', None)\n"
    "            if prev is not None:\n"
    "                m = min(int(prev.shape[0]), n)\n"
    "                agree = (prev[:m] == last_sampled[:m]).float().mean().item()\n"
    "                msg.append(f'pos0_agreement_prev_draft_vs_new_anchor={agree:.4f}')\n"
    "            d = self.draft_tokens[:n]\n"
    "            self._gv_prev_draft0 = d[:, 0].detach().clone()\n"
    "            msg.append(f'draft_row0={d[0].tolist()} last_sampled_head={last_sampled[:min(n, 8)].tolist()}')\n"
    "            print(' | '.join(msg), flush=True)\n"
    "        except Exception as e:\n"
    "            print('GV_DIAG2 error:', repr(e), flush=True)\n"
    "\n"
)

print(
    "diag:",
    patch(
        S,
        "        self._sample_after_forward(input_batch, num_sample_reqs)\n\n"
        "        return self.draft_tokens[:num_reqs]\n\n"
        "    def _sample_after_forward(self, input_batch: InputBatch, num_reqs: int) -> None:\n",
        "        self._sample_after_forward(input_batch, num_sample_reqs)\n\n"
        "        if os.environ.get('GV_DIAG') and not dummy_run and not is_profile:\n"
        "            self._gv_diag(num_reqs, aux_hidden_states, last_sampled)\n\n"
        "        return self.draft_tokens[:num_reqs]\n\n"
        + DIAG_METHOD +
        "    def _sample_after_forward(self, input_batch: InputBatch, num_reqs: int) -> None:\n",
    ),
)

import ast

ast.parse(open(S).read())
print("GV_DIAG_PATCH_OK")
