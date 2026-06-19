#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrast-domain masked comparison between OPD simulation and experiment.

Pipeline:
1. Load 128x128 generated OPD crops and the experimental particle stack.
2. Estimate one stack-level background from border pixels.
3. Convert both stacks to contrast images: C = I / bg - 1.
4. Estimate particle centers from high-contrast pixels and shift experimental
   frames to the generated particle center.
5. Build one foreground mask per generated depth from |C_gen|.
6. Compute masked SSIM, masked NCC, and masked contrast RMSE.
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
from scipy.ndimage import (
    binary_dilation,
    center_of_mass,
    gaussian_filter,
    label as ndi_label,
    shift as ndi_shift,
)


BASE_DIR = Path(__file__).resolve().parent
PREVIOUS_COMPARE_DIR = BASE_DIR / "opd_vs_experiment_314_particle_128_ssim"
GEN_CROP_DIR = PREVIOUS_COMPARE_DIR / "generated_opd_center_crop_128"
EXP_PATH = Path(
    "/Users/wangzhuofan/Desktop/Uchiyamalab/Experiment result/Image analysis/"
    "260615 2um beads exposure 2ms 100 intensity 73% intensity 20x 0.5 conc/"
    "Experiment-314 的单个粒子.tif"
)
OUT_DIR = BASE_DIR / "opd_vs_experiment_314_particle_128_contrast_masked"
CROP_SIZE = 128
Z_VALUES = list(range(-50, 51, 10))


def parse_z(path: Path) -> int:
    match = re.search(r"z_([+-]\d+)um", path.name)
    if not match:
        raise ValueError(f"Cannot parse z value from {path.name}")
    return int(match.group(1))


def load_generated_crop_stack():
    files = sorted(GEN_CROP_DIR.glob("opd_generated_crop128_z_*.tif"), key=parse_z)
    z_values = [parse_z(path) for path in files]
    if z_values != Z_VALUES:
        raise RuntimeError(f"Generated z list mismatch: {z_values}")
    stack = np.stack([tiff.imread(path).astype(np.float64) for path in files], axis=0)
    if stack.shape[1:] != (CROP_SIZE, CROP_SIZE):
        raise RuntimeError(f"Generated crop shape mismatch: {stack.shape}")
    return z_values, stack


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


def border_values(stack: np.ndarray, border_width: int = 12) -> np.ndarray:
    top = stack[:, :border_width, :]
    bottom = stack[:, -border_width:, :]
    left = stack[:, border_width:-border_width, :border_width]
    right = stack[:, border_width:-border_width, -border_width:]
    return np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])


def stack_contrast(stack: np.ndarray):
    bg = float(np.median(border_values(stack)))
    return stack / bg - 1.0, bg


def estimate_particle_center(contrast_image: np.ndarray):
    abs_contrast = np.abs(contrast_image)
    smoothed = gaussian_filter(abs_contrast, sigma=1.0, mode="reflect")
    threshold = max(0.03, float(np.quantile(smoothed, 0.985)))
    mask = smoothed > threshold
    if int(mask.sum()) < 20:
        threshold = float(np.quantile(smoothed, 0.99))
        mask = smoothed > threshold
    if int(mask.sum()) < 5:
        center = ((contrast_image.shape[0] - 1) / 2, (contrast_image.shape[1] - 1) / 2)
        return center, threshold, int(mask.sum())

    cy, cx = center_of_mass(smoothed * mask)
    return (float(cy), float(cx)), threshold, int(mask.sum())


def align_experiment_to_generated(exp_contrast, gen_contrast):
    gen_centers = []
    exp_centers = []
    shifts = []
    aligned = []
    center_rows = []

    target_center = np.array(
        [estimate_particle_center(image)[0] for image in gen_contrast],
        dtype=np.float64,
    ).mean(axis=0)

    for z, exp_image, gen_image in zip(Z_VALUES, exp_contrast, gen_contrast):
        gen_center, gen_thr, gen_n = estimate_particle_center(gen_image)
        exp_center, exp_thr, exp_n = estimate_particle_center(exp_image)

        shift_yx = target_center - np.array(exp_center, dtype=np.float64)
        shifted = ndi_shift(exp_image, shift=shift_yx, order=1, mode="nearest")

        gen_centers.append(gen_center)
        exp_centers.append(exp_center)
        shifts.append(tuple(float(v) for v in shift_yx))
        aligned.append(shifted)
        center_rows.append(
            [
                z,
                gen_center[1],
                gen_center[0],
                exp_center[1],
                exp_center[0],
                shift_yx[1],
                shift_yx[0],
                gen_thr,
                exp_thr,
                gen_n,
                exp_n,
            ]
        )

    return np.stack(aligned, axis=0), np.array(shifts), center_rows, target_center


def build_generated_masks(gen_contrast):
    masks = []
    rows = []
    for z, image in zip(Z_VALUES, gen_contrast):
        abs_contrast = gaussian_filter(np.abs(image), sigma=1.5, mode="reflect")
        p99 = float(np.quantile(abs_contrast, 0.99))
        threshold = max(0.015, 0.45 * p99)
        initial_mask = abs_contrast > threshold

        labels, n_labels = ndi_label(initial_mask)
        mask = np.zeros_like(initial_mask, dtype=bool)
        for label_id in range(1, n_labels + 1):
            component = labels == label_id
            cy, cx = center_of_mass(component)
            distance_from_center = np.hypot(cy - 64.0, cx - 64.0)
            if distance_from_center < 52.0 or component[60:69, 60:69].any():
                mask |= component

        mask = binary_dilation(mask, iterations=3)
        masks.append(mask)
        rows.append([z, threshold, int(mask.sum()), float(mask.sum() / mask.size)])
    return np.stack(masks, axis=0), rows


def local_ssim_map(x: np.ndarray, y: np.ndarray, data_range: float):
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
    return ((2.0 * uxuy + c1) * (2.0 * vxy + c2)) / (
        (ux2 + uy2 + c1) * (vx + vy + c2)
    )


def masked_ncc(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    xv = x[mask]
    yv = y[mask]
    xv = xv - float(np.mean(xv))
    yv = yv - float(np.mean(yv))
    denom = np.sqrt(float(np.sum(xv * xv)) * float(np.sum(yv * yv)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(xv * yv) / denom)


def compute_metrics(gen_contrast, exp_contrast_aligned, masks):
    n = len(Z_VALUES)
    data_min = min(float(np.min(gen_contrast)), float(np.min(exp_contrast_aligned)))
    data_max = max(float(np.max(gen_contrast)), float(np.max(exp_contrast_aligned)))
    data_range = data_max - data_min

    ssim = np.zeros((n, n), dtype=float)
    ncc = np.zeros((n, n), dtype=float)
    rmse = np.zeros((n, n), dtype=float)
    gen_energy = np.zeros(n, dtype=float)
    exp_energy = np.zeros((n, n), dtype=float)

    for i in range(n):
        mask = masks[i]
        gen_energy[i] = float(np.sqrt(np.mean(gen_contrast[i][mask] ** 2)))
        for j in range(n):
            ssim_map = local_ssim_map(gen_contrast[i], exp_contrast_aligned[j], data_range)
            ssim[i, j] = float(np.mean(ssim_map[mask]))
            ncc[i, j] = masked_ncc(gen_contrast[i], exp_contrast_aligned[j], mask)
            diff = gen_contrast[i][mask] - exp_contrast_aligned[j][mask]
            rmse[i, j] = float(np.sqrt(np.mean(diff * diff)))
            exp_energy[i, j] = float(
                np.sqrt(np.mean(exp_contrast_aligned[j][mask] ** 2))
            )

    return ssim, ncc, rmse, gen_energy, exp_energy, data_range


def write_matrix_csv(path: Path, matrix: np.ndarray):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generated_z_um/experiment_z_um", *Z_VALUES])
        for z, row in zip(Z_VALUES, matrix):
            writer.writerow([z, *[f"{v:.8f}" for v in row]])


def write_rows_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_stack_and_frames(stack, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for z, image in zip(Z_VALUES, stack):
        tiff.imwrite(out_dir / f"{prefix}_z_{z:+d}um.tif", image.astype(np.float32))
    tiff.imwrite(out_dir / f"{prefix}_stack.tif", stack.astype(np.float32))


def save_masks(masks, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for z, mask in zip(Z_VALUES, masks):
        tiff.imwrite(out_dir / f"generated_foreground_mask_z_{z:+d}um.tif", mask.astype(np.uint8) * 255)
    tiff.imwrite(out_dir / "generated_foreground_mask_stack.tif", masks.astype(np.uint8) * 255)


def plot_contrast_contact_sheet(gen_contrast, exp_contrast_aligned, masks, out_path: Path):
    limit = float(np.quantile(np.abs(np.concatenate([gen_contrast.ravel(), exp_contrast_aligned.ravel()])), 0.995))
    fig, axes = plt.subplots(3, len(Z_VALUES), figsize=(2.0 * len(Z_VALUES), 5.8), dpi=180)
    for col, z in enumerate(Z_VALUES):
        axes[0, col].imshow(gen_contrast[col], cmap="gray", vmin=-limit, vmax=limit)
        axes[0, col].set_title(f"{z:+d} um", fontsize=8)
        axes[0, col].set_axis_off()

        axes[1, col].imshow(exp_contrast_aligned[col], cmap="gray", vmin=-limit, vmax=limit)
        axes[1, col].set_axis_off()

        axes[2, col].imshow(masks[col], cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_axis_off()

    axes[0, 0].set_ylabel("Gen C", fontsize=9)
    axes[1, 0].set_ylabel("Exp C\naligned", fontsize=9)
    axes[2, 0].set_ylabel("Mask", fontsize=9)
    fig.suptitle("Contrast images and generated foreground masks", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(matrix, title, colorbar_label, out_path: Path, cmap="viridis", reverse=False):
    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=180)
    if reverse:
        cmap = "magma_r"
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(np.arange(len(Z_VALUES)))
    ax.set_yticks(np.arange(len(Z_VALUES)))
    ax.set_xticklabels([f"{z:+d}" for z in Z_VALUES], rotation=45, ha="right")
    ax.set_yticklabels([f"{z:+d}" for z in Z_VALUES])
    ax.set_xlabel("Experimental depth z (um), aligned")
    ax.set_ylabel("Generated OPD depth z (um)")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_same_depth_metrics(ssim, ncc, rmse, out_path: Path):
    diag_ssim = np.diag(ssim)
    diag_ncc = np.diag(ncc)
    diag_rmse = np.diag(rmse)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), dpi=180, sharex=True)
    axes[0].plot(Z_VALUES, diag_ssim, marker="o", color="#1f77b4")
    axes[0].set_ylabel("Masked SSIM")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(Z_VALUES, diag_ncc, marker="o", color="#2ca02c")
    axes[1].set_ylabel("Masked NCC")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(Z_VALUES, diag_rmse, marker="o", color="#d62728")
    axes[2].set_ylabel("Contrast RMSE")
    axes[2].set_xlabel("Depth z (um)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(Z_VALUES)

    fig.suptitle("Same-depth masked contrast metrics")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_matches(metric, metric_name, out_path: Path, maximize=True):
    if maximize:
        best_indices = np.argmax(metric, axis=1)
    else:
        best_indices = np.argmin(metric, axis=1)
    best_z = np.array([Z_VALUES[i] for i in best_indices])
    best_values = metric[np.arange(len(Z_VALUES)), best_indices]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    ax1.plot(Z_VALUES, best_z, marker="o", color="#1f77b4", label="Best experimental z")
    ax1.plot(Z_VALUES, Z_VALUES, linestyle="--", color="gray", linewidth=1.0, label="same-depth line")
    ax1.set_xlabel("Generated OPD depth z (um)")
    ax1.set_ylabel("Best matched experimental depth z (um)")
    ax1.set_xticks(Z_VALUES)
    ax1.set_yticks(Z_VALUES)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(Z_VALUES, best_values, marker="s", color="#d62728", label=f"Best {metric_name}")
    ax2.set_ylabel(metric_name)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8, frameon=False, loc="upper left")
    ax1.set_title(f"Best experimental match by {metric_name}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return best_z, best_values


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    z_values, gen_stack = load_generated_crop_stack()
    if z_values != Z_VALUES:
        raise RuntimeError("Unexpected z values")
    exp_raw = rgb_to_gray(tiff.imread(EXP_PATH))
    if exp_raw.shape != gen_stack.shape:
        raise RuntimeError(f"Shape mismatch: generated {gen_stack.shape}, experiment {exp_raw.shape}")

    gen_contrast, gen_bg = stack_contrast(gen_stack)
    exp_contrast, exp_bg = stack_contrast(exp_raw)
    exp_contrast_aligned, shifts, center_rows, target_center = align_experiment_to_generated(
        exp_contrast,
        gen_contrast,
    )
    masks, mask_rows = build_generated_masks(gen_contrast)

    ssim, ncc, rmse, gen_energy, exp_energy, data_range = compute_metrics(
        gen_contrast,
        exp_contrast_aligned,
        masks,
    )

    save_stack_and_frames(gen_contrast, OUT_DIR / "generated_contrast_128", "opd_generated_contrast")
    save_stack_and_frames(exp_contrast, OUT_DIR / "experiment_contrast_128", "experiment_314_contrast")
    save_stack_and_frames(exp_contrast_aligned, OUT_DIR / "experiment_contrast_aligned_128", "experiment_314_contrast_aligned")
    save_masks(masks, OUT_DIR / "generated_foreground_masks")

    write_matrix_csv(OUT_DIR / "masked_ssim_matrix.csv", ssim)
    write_matrix_csv(OUT_DIR / "masked_ncc_matrix.csv", ncc)
    write_matrix_csv(OUT_DIR / "masked_contrast_rmse_matrix.csv", rmse)

    write_rows_csv(
        OUT_DIR / "alignment_centers_and_shifts.csv",
        [
            "z_um",
            "generated_center_x",
            "generated_center_y",
            "experiment_center_x",
            "experiment_center_y",
            "applied_shift_x",
            "applied_shift_y",
            "generated_center_threshold",
            "experiment_center_threshold",
            "generated_center_pixels",
            "experiment_center_pixels",
        ],
        center_rows,
    )
    write_rows_csv(
        OUT_DIR / "generated_mask_summary.csv",
        ["z_um", "threshold_abs_contrast", "mask_pixels", "mask_fraction"],
        mask_rows,
    )

    same_rows = []
    for idx, z in enumerate(Z_VALUES):
        same_rows.append(
            [
                z,
                f"{ssim[idx, idx]:.8f}",
                f"{ncc[idx, idx]:.8f}",
                f"{rmse[idx, idx]:.8f}",
                f"{gen_energy[idx]:.8f}",
                f"{exp_energy[idx, idx]:.8f}",
            ]
        )
    write_rows_csv(
        OUT_DIR / "same_depth_masked_metrics.csv",
        [
            "z_um",
            "masked_ssim",
            "masked_ncc",
            "masked_contrast_rmse",
            "generated_masked_contrast_energy",
            "experiment_masked_contrast_energy",
        ],
        same_rows,
    )

    best_z_ssim, best_ssim = plot_best_matches(
        ssim,
        "Masked SSIM",
        OUT_DIR / "best_match_by_masked_ssim.png",
        maximize=True,
    )
    best_z_ncc, best_ncc = plot_best_matches(
        ncc,
        "Masked NCC",
        OUT_DIR / "best_match_by_masked_ncc.png",
        maximize=True,
    )
    best_z_rmse, best_rmse = plot_best_matches(
        rmse,
        "Contrast RMSE",
        OUT_DIR / "best_match_by_contrast_rmse.png",
        maximize=False,
    )

    best_rows = []
    for idx, z in enumerate(Z_VALUES):
        best_rows.append(
            [
                z,
                int(best_z_ssim[idx]),
                f"{best_ssim[idx]:.8f}",
                int(best_z_ncc[idx]),
                f"{best_ncc[idx]:.8f}",
                int(best_z_rmse[idx]),
                f"{best_rmse[idx]:.8f}",
            ]
        )
    write_rows_csv(
        OUT_DIR / "best_match_summary.csv",
        [
            "generated_z_um",
            "best_experiment_z_by_masked_ssim",
            "best_masked_ssim",
            "best_experiment_z_by_masked_ncc",
            "best_masked_ncc",
            "best_experiment_z_by_rmse",
            "best_rmse",
        ],
        best_rows,
    )

    plot_contrast_contact_sheet(
        gen_contrast,
        exp_contrast_aligned,
        masks,
        OUT_DIR / "contrast_and_mask_contact_sheet.png",
    )
    plot_metric_heatmap(
        ssim,
        "Masked SSIM on contrast images",
        "Masked SSIM",
        OUT_DIR / "masked_ssim_heatmap.png",
    )
    plot_metric_heatmap(
        ncc,
        "Masked NCC on contrast images",
        "Masked NCC",
        OUT_DIR / "masked_ncc_heatmap.png",
    )
    plot_metric_heatmap(
        rmse,
        "Masked contrast RMSE",
        "RMSE",
        OUT_DIR / "masked_contrast_rmse_heatmap.png",
        reverse=True,
    )
    plot_same_depth_metrics(ssim, ncc, rmse, OUT_DIR / "same_depth_masked_metrics.png")

    with (OUT_DIR / "comparison_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Contrast-domain masked OPD simulation vs Experiment-314 comparison\n")
        f.write(f"generated_crop_dir = {GEN_CROP_DIR}\n")
        f.write(f"experiment_path = {EXP_PATH}\n")
        f.write(f"z_values_um = {Z_VALUES}\n")
        f.write(f"generated_stack_background_median_border = {gen_bg:.12f}\n")
        f.write(f"experiment_stack_background_median_border = {exp_bg:.12f}\n")
        f.write("contrast_definition = I / stack_background - 1\n")
        f.write("mask_definition = gaussian(abs(C_gen), sigma=1.5) > max(0.015, 0.45 * p99), keep center-near components, dilated 3 px\n")
        f.write(f"target_center_yx = [{target_center[0]:.4f}, {target_center[1]:.4f}]\n")
        f.write(f"ssim_data_range = {data_range:.12f}\n\n")
        f.write("Same-depth metrics:\n")
        for row in same_rows:
            f.write(
                "  z={:+d} um: masked_ssim={}, masked_ncc={}, rmse={}, gen_energy={}, exp_energy={}\n".format(
                    int(row[0]),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                )
            )
        f.write("\nBest matches:\n")
        for row in best_rows:
            f.write(
                "  generated z={:+d} um: SSIM best exp z={:+d} ({:.8f}), NCC best exp z={:+d} ({:.8f}), RMSE best exp z={:+d} ({:.8f})\n".format(
                    int(row[0]),
                    int(row[1]),
                    float(row[2]),
                    int(row[3]),
                    float(row[4]),
                    int(row[5]),
                    float(row[6]),
                )
            )

    print("output_dir", OUT_DIR)
    print("generated_bg", gen_bg)
    print("experiment_bg", exp_bg)
    print("target_center_yx", target_center)
    print("same_depth_metrics")
    for row in same_rows:
        print(
            f"  z={int(row[0]):+d} ssim={row[1]} ncc={row[2]} rmse={row[3]} "
            f"genE={row[4]} expE={row[5]}"
        )
    print("best_match_by_masked_ssim")
    for z, bz, value in zip(Z_VALUES, best_z_ssim, best_ssim):
        print(f"  generated z={z:+d} -> experiment z={int(bz):+d}: {value:.8f}")


if __name__ == "__main__":
    main()
