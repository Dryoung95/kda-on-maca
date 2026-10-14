#!/usr/bin/env python3
"""D7 reproducer: large-tensor Conv1d silent data corruption
(MACA 3.3.0.15, MetaX C500).

Symptom: Conv1d(k=3, stride=3, dilation=4) over a (64, 64, 524280) input
returns garbage for the first 16 and the last 16 batches (|diff| up to ~3),
while the middle 32 batches are correct. The program does not crash and
reports no error.

Adjudication: no path is assumed guilty. An im2col unfold + GEMM reference
(which never runs the conv kernel) acts as a third-party umpire: per-batch
conv agrees with it to ~7e-7 on every batch, so only the full-tensor conv
path is wrong at large scale.

Trigger (bisected with B fixed at 64, sweeping L):
    in_elems < 1.622e9 (L=396000)     -> ok
    in_elems > 1.638e9 (L=400000)     -> BAD
Number of bad batches at full L is exactly 2 * (B - 48): the first (B-48)
and the last (B-48).

Run inside the MACA PyTorch environment:
    python3 conv1d_large_tensor.py
"""
import gc

import torch
import torch.nn.functional as F

C_IN, C_OUT, K, STRIDE, DIL = 64, 128, 3, 3, 4


def ref_im2col(conv, xb):
    """Third-party umpire: im2col + GEMM, never runs the conv kernel."""
    cols = F.unfold(xb.unsqueeze(3), kernel_size=(K, 1), dilation=(DIL, 1),
                    stride=(STRIDE, 1))
    w2 = conv.weight.detach().reshape(C_OUT, -1)
    return torch.einsum("ok,bkl->bol", w2, cols)


def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(42)
    conv = torch.nn.Conv1d(C_IN, C_OUT, K, stride=STRIDE, dilation=DIL,
                           bias=False).to(dev).eval()
    torch.backends.cudnn.deterministic = True   # has no effect on this defect

    # 1. Validate the umpire against conv on a small tensor first.
    torch.manual_seed(7)
    small = torch.rand(2, C_IN, 4096, device=dev)
    with torch.no_grad():
        err = (conv(small) - ref_im2col(conv, small)).abs().max().item()
    print(f"umpire validation (small tensor): conv vs im2col maxdiff = {err:.3e}")
    assert err < 1e-4, "umpire disagrees with conv; adjudication is invalid"
    del small
    gc.collect(); torch.cuda.empty_cache()

    # 2. Full size.
    torch.manual_seed(123)
    x = torch.rand(64, C_IN, 524280, device=dev)
    print(f"input (64, {C_IN}, 524280) = {x.numel():,} elements "
          f"(int32 ceiling {2**31:,}, margin {2**31 - x.numel():,})")

    probes = [0, 1, 16, 31, 47, 48, 62, 63]
    with torch.no_grad():
        full = conv(x)
        keep = full[probes].clone()
    del full
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'batch':>5}  {'|per-batch - umpire|':>21}  {'|full - umpire|':>17}")
    with torch.no_grad():
        for i, b in enumerate(probes):
            truth = ref_im2col(conv, x[b:b + 1])
            per = conv(x[b:b + 1])
            err_per = (per - truth).abs().max().item()
            err_full = (keep[i] - truth).abs().max().item()
            verdict = "ok" if err_full < 1e-4 else "GARBAGE"
            print(f"{b:>5}  {err_per:21.3e}  {err_full:17.3e}  {verdict}")
            del truth, per
            gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
