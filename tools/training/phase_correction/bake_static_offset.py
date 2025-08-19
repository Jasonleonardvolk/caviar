from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\tools\training\phase_correction\bake_static_offset.py
# Usage (Windows PowerShell):
#   (tori-kha-py3.11) PS {PROJECT_ROOT}> `
#   python tools/training/phase_correction/bake_static_offset.py `
#     --artifact_glob "data\calib\ios_a17\artifact_*.npy" `
#     --clean_glob    "data\calib\ios_a17\clean_*.npy" `
#     --out_dir       "public\corrections\ios_a17" `
#     --iters 600 --tv_lambda 0.08 --phase_w 1.0 --mag_w 0.2 --max_correction 0.30
#
# Linux/macOS:
#   python tools/training/phase_correction/bake_static_offset.py \
#     --artifact_glob data/calib/ios_a17/artifact_*.npy \
#     --clean_glob    data/calib/ios_a17/clean_*.npy \
#     --out_dir       public/corrections/ios_a17 \
#     --iters 600 --tv_lambda 0.08 --phase_w 1.0 --mag_w 0.2 --max_correction 0.30

import argparse, glob, json, os, time
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

def _as_complex_torch(a: np.ndarray) -> torch.Tensor:
    """
    Accepts:
      - complex64/complex128 [H,W]
      - [2,H,W] real/imag
      - [H,W,2] real/imag
    Returns: torch.complex64 [H,W]
    """
    if np.iscomplexobj(a):
        z = torch.from_numpy(a.astype(np.complex64))
        return z
    if a.ndim == 3 and a.shape[0] == 2:
        re = torch.from_numpy(a[0].astype(np.float32))
        im = torch.from_numpy(a[1].astype(np.float32))
        return torch.complex(re, im)
    if a.ndim == 3 and a.shape[-1] == 2:
        re = torch.from_numpy(a[...,0].astype(np.float32))
        im = torch.from_numpy(a[...,1].astype(np.float32))
        return torch.complex(re, im)
    raise ValueError(f"Unsupported array shape/dtype for complex: {a.shape} {a.dtype}")

@torch.no_grad()
def _save_bin_and_meta(dphi: torch.Tensor, out_dir: str, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    d = dphi.detach().cpu().to(torch.float32).numpy()  # [H,W]
    bin_path = os.path.join(out_dir, "phase_offset_f32.bin")
    with open(bin_path, "wb") as f:
        d.tofile(f)
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Wrote {bin_path} ({d.nbytes/1e6:.2f} MB)")
    print(f"[OK] Wrote {meta_path}")

def phasor_phase_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """ pred/tgt complex [...]; loss = mean(1 - cos(Dphi)) """
    dphi = torch.atan2(pred.imag, pred.real) - torch.atan2(tgt.imag, tgt.real)
    return (1.0 - torch.cos(dphi)).mean()

def mag_l1_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(torch.abs(pred), torch.abs(tgt))

def tv_l1(x: torch.Tensor) -> torch.Tensor:
    """ x [1,1,H,W] """
    dx = torch.diff(x, dim=-2)
    dy = torch.diff(x, dim=-1)
    return dx.abs().mean() + dy.abs().mean()

def load_pairs(artifact_glob: str, clean_glob: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    apaths = sorted(glob.glob(artifact_glob))
    cpaths = sorted(glob.glob(clean_glob))
    if not apaths or not cpaths:
        raise FileNotFoundError("No files found. Check globs.")
    if len(apaths) != len(cpaths):
        raise ValueError(f"Mismatched counts: {len(apaths)} artifacts vs {len(cpaths)} clean")
    arts, cleans = [], []
    for a_path, c_path in zip(apaths, cpaths):
        a = np.load(a_path)
        c = np.load(c_path)
        az = _as_complex_torch(a)
        cz = _as_complex_torch(c)
        if az.shape != cz.shape:
            raise ValueError(f"Shape mismatch: {a_path} {az.shape} vs {c_path} {cz.shape}")
        arts.append(az)
        cleans.append(cz)
    return arts, cleans

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_glob", required=True)
    p.add_argument("--clean_glob", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.25)
    p.add_argument("--phase_w", type=float, default=1.0)
    p.add_argument("--mag_w", type=float, default=0.2)
    p.add_argument("--tv_lambda", type=float, default=0.08)
    p.add_argument("--max_correction", type=float, default=0.30)
    p.add_argument("--downsample", type=int, default=1, help="optional integer downsample")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arts, cleans = load_pairs(args.artifact_glob, args.clean_glob)
    H, W = arts[0].shape[-2], arts[0].shape[-1]
    if args.downsample > 1:
        ds = args.downsample
        H //= ds; W //= ds
        arts  = [a[..., ::ds, ::ds] for a in arts]
        cleans = [c[..., ::ds, ::ds] for c in cleans]

    # Stack to [N,H,W] complex64
    A = torch.stack(arts, dim=0).to(device)
    C = torch.stack(cleans, dim=0).to(device)
    N = A.shape[0]

    # Learn a single Dphi map [1,1,H,W] shared across all samples
    dphi = torch.zeros((1,1,H,W), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([dphi], lr=args.lr, amsgrad=True)

    maxc = float(args.max_correction)
    phase_w = float(args.phase_w)
    mag_w   = float(args.mag_w)
    tv_w    = float(args.tv_lambda)

    print(f"[bake] N={N} HxW={H}x{W} lr={args.lr} iters={args.iters} "
          f"| weights: phase={phase_w} mag={mag_w} tv={tv_w} | clamp={maxc:.3f} rad")

    t0 = time.time()
    for it in range(1, args.iters+1):
        opt.zero_grad(set_to_none=True)

        # Apply correction to all samples
        dphi_clamped = dphi.clamp(-maxc, maxc)                  # [1,1,H,W]
        # Broadcast to [N,H,W] and build phasor
        phi = dphi_clamped[0,0]
        corr = torch.polar(torch.ones_like(phi, device=device), phi)  # complex64 [H,W]
        corr = corr.unsqueeze(0).expand(N, H, W)

        pred = A * corr  # complex64 [N,H,W]

        loss_phase = phasor_phase_loss(pred, C)
        loss_mag   = mag_l1_loss(pred, C)
        loss_tv    = tv_l1(dphi_clamped)

        loss = phase_w*loss_phase + mag_w*loss_mag + tv_w*loss_tv
        loss.backward()

        # Optional gradient clip for stability
        torch.nn.utils.clip_grad_norm_([dphi], max_norm=1.0)

        opt.step()

        # Keep parameter bounded after step
        with torch.no_grad():
            dphi.clamp_(-maxc, maxc)

        if it % 50 == 0 or it == 1:
            dt = time.time() - t0
            print(f"[{it:04d}] loss={loss.item():.6f} "
                  f"(phase={loss_phase.item():.6f}, mag={loss_mag.item():.6f}, tv={loss_tv.item():.6f})  t={dt:.1f}s")

    # Save outputs
    with torch.no_grad():
        final_map = dphi[0,0].clamp(-maxc, maxc)

    meta = {
        "shape_hw": [int(H), int(W)],
        "N_samples": int(N),
        "iters": int(args.iters),
        "lr": args.lr,
        "weights": {"phase": phase_w, "mag": mag_w, "tv": tv_w},
        "max_correction": maxc,
        "artifact_glob": args.artifact_glob,
        "clean_glob": args.clean_glob,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_bin_and_meta(final_map, args.out_dir, meta)

if __name__ == "__main__":
    main()
