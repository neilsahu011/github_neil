"""
SVD Image Compression
---------------------
Compress a grayscale or RGB image using a rank-k SVD approximation.

Examples:
    python svd_compression.py image.jpg
    python svd_compression.py image.jpg --ks 5 20 50
    python svd_compression.py image.jpg --method numpy --ks 5 20 50 --no-show
    python svd_compression.py image.jpg --method scratch --ks 5 20 --max-size 128 --no-show
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. SVD FROM SCRATCH: POWER ITERATION + DEFLATION
# ---------------------------------------------------------------------------

def power_iteration(A, max_iter=300, tol=1e-7, rng=None):
    """Approximate the strongest singular triplet (u, sigma, v) of matrix A."""
    if rng is None:
        rng = np.random.default_rng(0)

    v = rng.standard_normal(A.shape[1])
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.zeros(A.shape[0]), 0.0, v
    v /= norm_v

    for _ in range(max_iter):
        w = A.T @ (A @ v)
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            break
        w /= norm_w

        # Singular vectors have an arbitrary sign; compare both signs.
        if np.linalg.norm(w - v) < tol or np.linalg.norm(w + v) < tol:
            v = w if np.linalg.norm(w - v) < tol else -w
            break
        v = w

    Av = A @ v
    sigma = np.linalg.norm(Av)
    u = Av / sigma if sigma > 0 else np.zeros(A.shape[0])
    return u, sigma, v


def svd_scratch(A, k, verbose=True):
    """
    Find up to k singular components using power iteration and deflation.
    This is an educational implementation; NumPy's SVD is more robust.
    """
    R = A.astype(float).copy()
    m, n = R.shape
    k = min(k, m, n)

    U = np.zeros((m, k))
    S = np.zeros(k)
    Vt = np.zeros((k, n))
    rng = np.random.default_rng(0)

    for i in range(k):
        u, sigma, v = power_iteration(R, rng=rng)
        if sigma < 1e-10:
            U, S, Vt = U[:, :i], S[:i], Vt[:i, :]
            break

        U[:, i] = u
        S[i] = sigma
        Vt[i, :] = v
        R -= sigma * np.outer(u, v)

        if verbose and (i + 1) % 10 == 0:
            print(f"    computed {i + 1}/{k} singular values", end="\r")

    if verbose:
        print(" " * 60, end="\r")
    return U, S, Vt


def svd_numpy(A, k):
    """Reference implementation using NumPy's built-in SVD."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k], S[:k], Vt[:k, :]


# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------

def load_image(path, color=False, max_size=512):
    """Load image as a float array, shrinking large images for speed."""
    img = Image.open(path).convert("RGB" if color else "L")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    return np.asarray(img, dtype=float)


def rebuild(U, S, Vt, k):
    """Reconstruct a matrix using its first k singular components."""
    k = min(k, len(S))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


def to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def psnr(original, approx):
    mse = np.mean((original - approx) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def compression_ratio(m, n, k):
    """Estimated original values / values stored by rank-k factors (one channel)."""
    return (m * n) / (k * (m + n + 1))


def main():
    parser = argparse.ArgumentParser(description="SVD image compression")
    parser.add_argument("image", nargs="?", help="path to image (jpg/png/etc.)")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 20, 50, 100],
                        help="ranks to compare, e.g. --ks 5 20 50 100")
    parser.add_argument("--method", choices=["scratch", "numpy"], default="scratch",
                        help="'scratch' = power iteration + deflation; 'numpy' = np.linalg.svd")
    parser.add_argument("--color", action="store_true", help="process RGB instead of grayscale")
    parser.add_argument("--max-size", type=int, default=512,
                        help="maximum length of the image's longest side")
    parser.add_argument("--out", default=None, help="output folder (default: svd_output)")
    parser.add_argument("--no-show", action="store_true", help="save files without opening plot windows")
    args = parser.parse_args()

    if not args.image:
        parser.error("Please provide an image path, e.g. python svd_compression.py image.jpg")

    if not os.path.isfile(args.image):
        sys.exit(f"Could not find image: {args.image}")

    img = load_image(args.image, color=args.color, max_size=args.max_size)
    img3 = img[:, :, None] if img.ndim == 2 else img
    m, n, nch = img3.shape

    ks = sorted(set(k for k in args.ks if k >= 1 and k <= min(m, n)))
    if not ks:
        sys.exit(f"All k values must be between 1 and {min(m, n)}.")
    kmax = max(ks)

    print(f"Image      : {os.path.basename(args.image)} ({m} x {n}, {'RGB' if nch == 3 else 'grayscale'})")
    print(f"Method     : {args.method}")
    print(f"Ranks (k)  : {ks}\n")

    svd_fn = svd_scratch if args.method == "scratch" else svd_numpy
    factors = []
    start = time.time()

    for c in range(nch):
        print(f"Computing SVD for channel {c + 1}/{nch} ...")
        factors.append(svd_fn(img3[:, :, c], kmax))

    print(f"SVD done in {time.time() - start:.1f} s\n")

    def reconstruct_image(k):
        channels = [rebuild(*factor, k) for factor in factors]
        return np.clip(np.stack(channels, axis=2), 0, 255)

    total_energy = np.sum(img3 ** 2)
    print(f"{'k':>5} {'ratio':>8} {'rel. error':>11} {'PSNR (dB)':>10} {'energy %':>10}")
    print("-" * 50)

    results = {}
    for k in ks:
        approx = reconstruct_image(k)
        rel_error = np.linalg.norm(img3 - approx) / max(np.linalg.norm(img3), 1e-12)
        retained_energy = (
            sum(np.sum(factor[1][:k] ** 2) for factor in factors)
            / max(total_energy, 1e-12) * 100
        )
        ratio = compression_ratio(m, n, k)
        results[k] = approx
        print(f"{k:>5} {ratio:>7.1f}x {rel_error:>11.4f} {psnr(img3, approx):>10.2f} {retained_energy:>9.2f}%")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(script_dir, "svd_output")
    os.makedirs(out_dir, exist_ok=True)

    def as_pil(arr3):
        arr = to_uint8(arr3)
        return Image.fromarray(arr[:, :, 0] if nch == 1 else arr)

    as_pil(img3).save(os.path.join(out_dir, "original.png"))
    for k, approx in results.items():
        as_pil(approx).save(os.path.join(out_dir, f"compressed_k{k}.png"))

    # Figure 1: side-by-side image comparison.
    cols = len(ks) + 1
    fig1, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.6))
    axes = np.atleast_1d(axes)
    cmap = "gray" if nch == 1 else None

    def show(ax, arr3, title):
        arr = to_uint8(arr3)
        ax.imshow(arr[:, :, 0] if nch == 1 else arr, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    show(axes[0], img3, f"Original\n{m*n*nch:,} values")
    for ax, k in zip(axes[1:], ks):
        stored = k * (m + n + 1) * nch
        show(ax, results[k], f"k = {k}\n{stored:,} factor values")
    fig1.suptitle(f"SVD image compression ({args.method})")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "comparison.png"), dpi=150)

    # Figure 2: analysis plots.
    curve_k = np.unique(np.linspace(1, kmax, min(30, kmax)).astype(int))
    errors, psnrs, energies = [], [], []

    for k in curve_k:
        approx = reconstruct_image(int(k))
        errors.append(np.linalg.norm(img3 - approx) / max(np.linalg.norm(img3), 1e-12))
        psnrs.append(psnr(img3, approx))
        energies.append(
            sum(np.sum(factor[1][:k] ** 2) for factor in factors)
            / max(total_energy, 1e-12) * 100
        )

    fig2, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    for c, factor in enumerate(factors):
        label = "gray" if nch == 1 else "RGB"[c]
        a1.semilogy(np.arange(1, len(factor[1]) + 1), factor[1], label=label)
    a1.set_title("Singular values (log scale)")
    a1.set_xlabel("Index i")
    a1.set_ylabel("Singular value")
    a1.grid(alpha=0.3)
    a1.legend()

    a2.plot(curve_k, energies, marker="o")
    a2.axhline(95, linestyle="--", linewidth=1, label="95%")
    a2.axhline(99, linestyle="--", linewidth=1, label="99%")
    a2.set_title("Energy retained vs k")
    a2.set_xlabel("k")
    a2.set_ylabel("Energy (%)")
    a2.grid(alpha=0.3)
    a2.legend()

    a3.plot(curve_k, errors, marker="o", label="Relative error")
    a3.set_xlabel("k")
    a3.set_ylabel("Relative Frobenius error")
    a3.set_title("Reconstruction quality vs k")
    a3.grid(alpha=0.3)
    a3b = a3.twinx()
    a3b.plot(curve_k, psnrs, marker="s", label="PSNR")
    a3b.set_ylabel("PSNR (dB)")

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "analysis.png"), dpi=150)

    if args.method == "scratch":
        reference = np.linalg.svd(img3[:, :, 0], compute_uv=False)[:min(10, kmax)]
        mine = factors[0][1][:len(reference)]
        print("Check vs NumPy (first channel, top singular values):")
        print("  scratch:", np.round(mine, 3))
        print("  NumPy  :", np.round(reference, 3))
        print()

    print(f"Saved everything to: {out_dir}")
    print("Files: original.png, compressed_k*.png, comparison.png, analysis.png")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
