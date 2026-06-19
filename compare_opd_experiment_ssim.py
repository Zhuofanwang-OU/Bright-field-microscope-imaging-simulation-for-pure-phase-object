#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare 128x128 OPD simulation crops with an experimental particle stack.

Outputs:
- cropped generated OPD TIFFs and stack
- background-aligned experimental TIFF stack
- same-depth SSIM CSV/plot
- generated-depth vs all experimental depths SSIM matrix CSV/heatmap/curves
"""

import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter


BASE_DIR = Path(__file__).resolve().parent
GEN_DIR = BASE_DIR / "results_tif_opd_led_sos_raw_grid256_pad2_NAill009_zstep10"
EXP_PATH = Path(
    "/Users/wangzhuofan/Desktop/Uchiyamalab/Experiment result/Image analysis/"
    "260615 2um beads exposure 2ms 100 intensity 73% intensity 20x 0.5 conc/"
    "Experiment-314 的单个粒子.tif"
)
OUT_DIR = BASE_DIR / "opd_vs_experiment_314_particle_128_ssim"
CROP_SIZE = 128
Z_VALUES = list(range(-50, 51, 10))


def parse_z(path: Path) -> int:
    match = re.search(r"z_([+-]\d+)um", path.name)
    if not match:
        raise ValueError(f"Cannot parse z value from {path.name}")
    return int(match.group(1))


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size]


def load_generated_stack():
    files = sorted(GEN_DIR.glob("opd_led_sos_raw_z_*.tif"), key=parse_z)
    z_values = [parse_z(path) for path in files]
    if z_values != Z_VALUES:
        raise RuntimeError(f"Generated z list mismatch: {z_values}")

    crops = []
    for path in files:
        image = tiff.imread(path).astype(np.float64)
        crops.append(center_crop(image, CROP_SIZE))
    return z_values, np.stack(crops, axis=0)


def rgb_to_gray(stack: np.ndarray) -> np.ndarray:
    stack = stack.astype(np.float64)
    if stack.ndim == 4 and stack.shape[-1] == 3:
        return (
            0.2126 * stack[..., 0]
            + 0.7152 * stack[..., 1]
            + 0.0722 * stack[..., 2]
        )
    if stack.ndim == 3:
        return stack
    raise RuntimeError(f"Unsupported experimental TIFF shape: {stack.shape}")


def border_pixels(stack: np.ndarray, border_width: int = 12) -> np.ndarray:
    top = stack[:, :border_width, :]
    bottom = stack[:, -border_width:, :]
    left = stack[:, border_width:-border_width, :border_width]
    right = stack[:, border_width:-border_width, -border_width:]
    return np.concatenate(
        [top.ravel(), bottom.ravel(), left.ravel(), right.ravel()]
    )


def save_stack_and_frames(stack: np.ndarray, z_values, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for z, image in zip(z_values, stack):
        tiff.imwrite(out_dir / f"{prefix}_z_{z:+d}um.tif", image.astype(np.float32))
    tiff.imwrite(out_dir / f"{prefix}_stack.tif", stack.astype(np.float32))


def local_ssim(x: np.ndarray, y: np.ndarray, data_range: float) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    sigma = 1.5
    ux = gaussian_filter(x, sigma=sigma, mode="reflect")
    uy = gaussian_filter(y, sigma=sigma, mode="reflect")

    ux2 = ux * ux
    uy2 = uy * uy
    uxuy = ux * uy

    vx = gaussian_filter(x * x, sigma=sigma, mode="reflect") - ux2
    vy = gaussian_filter(y * y, sigma=sigma, mode="reflect") - uy2
    vxy = gaussian_filter(x * y, sigma=sigma, mode="reflect") - uxuy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * uxuy + c1) * (2.0 * vxy + c2)
    denominator = (ux2 + uy2 + c1) * (vx + vy + c2)
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def write_matrix_csv(path: Path, z_values, matrix: np.ndarray):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generated_z_um/experiment_z_um", *z_values])
        for z, row in zip(z_values, matrix):
            writer.writerow([z, *[f"{v:.8f}" for v in row]])


def plot_contact_sheet(gen_stack, exp_stack, z_values, out_path: Path):
    vmin = min(float(np.min(gen_stack)), float(np.min(exp_stack)))
    vmax = max(float(np.max(gen_stack)), float(np.max(exp_stack)))
    fig, axes = plt.subplots(2, len(z_values), figsize=(2.0 * len(z_values), 4.2), dpi=180)

    for col, z in enumerate(z_values):
        axes[0, col].imshow(gen_stack[col], cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{z:+d} um", fontsize=8)
        axes[0, col].set_axis_off()
        axes[1, col].imshow(exp_stack[col], cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, col].set_axis_off()

    axes[0, 0].set_ylabel("OPD sim", fontsize=9)
    axes[1, 0].set_ylabel("Experiment", fontsize=9)
    fig.suptitle("128x128 center crops, common intensity scale", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_same_depth(z_values, diagonal, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    ax.plot(z_values, diagonal, marker="o", linewidth=1.8)
    ax.set_xlabel("Depth z (um)")
    ax.set_ylabel("SSIM")
    ax.set_title("Same-depth SSIM: OPD simulation vs experiment")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(z_values)
    for z, value in zip(z_values, diagonal):
        ax.annotate(f"{value:.3f}", (z, value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_matrix(z_values, matrix, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=180)
    im = ax.imshow(matrix, cmap="viridis", vmin=max(0.0, float(np.min(matrix))), vmax=float(np.max(matrix)))
    ax.set_xticks(np.arange(len(z_values)))
    ax.set_yticks(np.arange(len(z_values)))
    ax.set_xticklabels([f"{z:+d}" for z in z_values], rotation=45, ha="right")
    ax.set_yticklabels([f"{z:+d}" for z in z_values])
    ax.set_xlabel("Experimental depth z (um)")
    ax.set_ylabel("Generated OPD depth z (um)")
    ax.set_title("SSIM matrix: each generated depth vs all experimental depths")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")

    fig.colorbar(im, ax=ax, label="SSIM")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_depth_curves(z_values, matrix, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=180)
    cmap = plt.get_cmap("coolwarm", len(z_values))
    for idx, z in enumerate(z_values):
        ax.plot(
            z_values,
            matrix[idx],
            marker="o",
            linewidth=1.1,
            color=cmap(idx),
            label=f"gen {z:+d} um",
        )
    ax.set_xlabel("Experimental depth z (um)")
    ax.set_ylabel("SSIM")
    ax.set_title("For each generated depth, compare against all experimental depths")
    ax.set_xticks(z_values)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_matches(z_values, matrix, out_path: Path):
    best_indices = np.argmax(matrix, axis=1)
    best_z = np.array([z_values[i] for i in best_indices])
    best_ssim = matrix[np.arange(len(z_values)), best_indices]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    ax1.plot(z_values, best_z, marker="o", color="#1f77b4", label="Best experimental z")
    ax1.plot(z_values, z_values, linestyle="--", color="gray", linewidth=1.0, label="same-depth line")
    ax1.set_xlabel("Generated OPD depth z (um)")
    ax1.set_ylabel("Best matched experimental depth z (um)")
    ax1.set_xticks(z_values)
    ax1.set_yticks(z_values)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(z_values, best_ssim, marker="s", color="#d62728", label="Best SSIM")
    ax2.set_ylabel("Best SSIM")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8, frameon=False, loc="upper left")
    ax1.set_title("Best experimental match for each generated depth")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    z_values, gen_crop = load_generated_stack()
    exp_raw = tiff.imread(EXP_PATH)
    exp_gray = rgb_to_gray(exp_raw)
    if exp_gray.shape[0] != len(z_values):
        raise RuntimeError(f"Experimental stack has {exp_gray.shape[0]} frames, expected {len(z_values)}")
    if exp_gray.shape[1:] != (CROP_SIZE, CROP_SIZE):
        raise RuntimeError(f"Experimental frame size is {exp_gray.shape[1:]}, expected {(CROP_SIZE, CROP_SIZE)}")

    gen_bg = float(np.median(border_pixels(gen_crop)))
    exp_bg = float(np.median(border_pixels(exp_gray)))
    exp_aligned = exp_gray * (gen_bg / exp_bg)

    save_stack_and_frames(
        gen_crop,
        z_values,
        OUT_DIR / "generated_opd_center_crop_128",
        "opd_generated_crop128",
    )
    save_stack_and_frames(
        exp_aligned,
        z_values,
        OUT_DIR / "experiment_background_aligned_128",
        "experiment_314_bg_aligned",
    )

    data_min = min(float(np.min(gen_crop)), float(np.min(exp_aligned)))
    data_max = max(float(np.max(gen_crop)), float(np.max(exp_aligned)))
    data_range = data_max - data_min

    ssim_matrix = np.zeros((len(z_values), len(z_values)), dtype=float)
    for i in range(len(z_values)):
        for j in range(len(z_values)):
            ssim_matrix[i, j] = local_ssim(gen_crop[i], exp_aligned[j], data_range)

    diagonal = np.diag(ssim_matrix)
    best_indices = np.argmax(ssim_matrix, axis=1)
    best_z = [z_values[i] for i in best_indices]
    best_ssim = ssim_matrix[np.arange(len(z_values)), best_indices]

    write_matrix_csv(OUT_DIR / "ssim_matrix_generated_vs_experiment.csv", z_values, ssim_matrix)

    with (OUT_DIR / "same_depth_ssim.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["z_um", "ssim_same_depth"])
        for z, value in zip(z_values, diagonal):
            writer.writerow([z, f"{value:.8f}"])

    with (OUT_DIR / "best_match_by_generated_depth.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generated_z_um", "best_experiment_z_um", "best_ssim"])
        for z, matched_z, value in zip(z_values, best_z, best_ssim):
            writer.writerow([z, matched_z, f"{value:.8f}"])

    plot_contact_sheet(gen_crop, exp_aligned, z_values, OUT_DIR / "crop_contact_sheet_common_scale.png")
    plot_same_depth(z_values, diagonal, OUT_DIR / "same_depth_ssim.png")
    plot_matrix(z_values, ssim_matrix, OUT_DIR / "ssim_matrix_heatmap.png")
    plot_all_depth_curves(z_values, ssim_matrix, OUT_DIR / "ssim_all_depth_curves.png")
    plot_best_matches(z_values, ssim_matrix, OUT_DIR / "best_match_by_generated_depth.png")

    with (OUT_DIR / "comparison_summary.txt").open("w", encoding="utf-8") as f:
        f.write("OPD simulation vs Experiment-314 particle SSIM comparison\n")
        f.write(f"generated_dir = {GEN_DIR}\n")
        f.write(f"experiment_path = {EXP_PATH}\n")
        f.write(f"z_values_um = {z_values}\n")
        f.write(f"crop_size_px = {CROP_SIZE}\n")
        f.write(f"generated_background_median_border = {gen_bg:.12f}\n")
        f.write(f"experimental_background_median_border_raw = {exp_bg:.12f}\n")
        f.write(f"experimental_scale_factor = {gen_bg / exp_bg:.12f}\n")
        f.write(f"ssim_data_range = {data_range:.12f}\n")
        f.write("\nSame-depth SSIM:\n")
        for z, value in zip(z_values, diagonal):
            f.write(f"  z={z:+d} um: {value:.8f}\n")
        f.write("\nBest experimental match for each generated depth:\n")
        for z, matched_z, value in zip(z_values, best_z, best_ssim):
            f.write(f"  generated z={z:+d} um -> experiment z={matched_z:+d} um: {value:.8f}\n")

    print("output_dir", OUT_DIR)
    print("generated_background", gen_bg)
    print("experimental_background_raw", exp_bg)
    print("experimental_scale_factor", gen_bg / exp_bg)
    print("same_depth_ssim")
    for z, value in zip(z_values, diagonal):
        print(f"  z={z:+d} {value:.8f}")
    print("best_matches")
    for z, matched_z, value in zip(z_values, best_z, best_ssim):
        print(f"  generated z={z:+d} -> experiment z={matched_z:+d}: {value:.8f}")


if __name__ == "__main__":
    main()
