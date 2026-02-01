#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:09:21 2025

@author: wangzhuofan
"""

import tifffile as tiff
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def simulate_partial_coherent_imaging():
    # ================= 参数 =================
    wavelength = 0.550       # um
    pixel_size = 0.05        # um
    grid_size = 512

    # bead
    bead_radius = 0.5
    n_bead = 1.59
    n_medium = 1.50

    # objective
    NA_obj = 0.16
    z_list = [-30, -20, -10, 0, 10, 20, 30]

    # illumination (partial coherence)
    NA_ill = 0.09           # illumination NA
    N_src = 9               # source grid per dimension

    eps = 1e-12             # 防止除零

    # ================= 1. object =================
    x = (np.arange(grid_size) - grid_size/2) * pixel_size
    y = (np.arange(grid_size) - grid_size/2) * pixel_size
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    thickness = 2 * np.sqrt(np.maximum(0.0, bead_radius**2 - R**2))
    k0 = 2 * np.pi / wavelength
    phase_shift = k0 * (n_bead - n_medium) * thickness
    O = np.exp(1j * phase_shift)

    # ================= 2. object spectrum =================
    O_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(O)))

    fx = np.fft.fftshift(np.fft.fftfreq(grid_size, d=pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(grid_size, d=pixel_size))
    FX, FY = np.meshgrid(fx, fy)
    F_sq = FX**2 + FY**2

    # frequency interpolator (complex)
    interp_real = RegularGridInterpolator((fy, fx), O_f.real, bounds_error=False, fill_value=0.0)
    interp_imag = RegularGridInterpolator((fy, fx), O_f.imag, bounds_error=False, fill_value=0.0)

    # ====== 2.5 flat-field spectrum (O0 = 1) ======
    # 用于对每个 source 点做通光量归一化，去掉大尺度包络
    O0 = np.ones((grid_size, grid_size), dtype=complex)
    O0_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(O0)))

    interp0_real = RegularGridInterpolator((fy, fx), O0_f.real, bounds_error=False, fill_value=0.0)
    interp0_imag = RegularGridInterpolator((fy, fx), O0_f.imag, bounds_error=False, fill_value=0.0)

    # ================= 3. pupil =================
    f_c = NA_obj / wavelength
    pupil_amp = (F_sq <= f_c**2).astype(float)

    # ================= 4. illumination source =================
    src_lin = np.linspace(-NA_ill, NA_ill, N_src)
    xi_grid, eta_grid = np.meshgrid(src_lin, src_lin)
    src_mask = (xi_grid**2 + eta_grid**2) <= NA_ill**2

    xi_list = xi_grid[src_mask]
    # xi_list = np.array([0.0])
    eta_list = eta_grid[src_mask]
    # eta_list = np.array([0.0])

    # uniform source
    S_src = np.ones_like(xi_list, dtype=float)
    S_src /= S_src.sum()
    # S_src = np.array([1.0])
    # ================= 5. imaging =================
    images = []

    for z in z_list:
        pupil_phase = np.exp(1j * np.pi * wavelength * z * F_sq)
        P = pupil_amp * pupil_phase

        I_sum = np.zeros((grid_size, grid_size), dtype=float)

        for xi, eta, w in zip(xi_list, eta_list, S_src):
            fx_shift = FX - xi / wavelength
            fy_shift = FY - eta / wavelength
            pts = np.stack([fy_shift.ravel(), fx_shift.ravel()], axis=-1)

            # shifted object spectrum
            O_shift = (
                interp_real(pts).reshape(grid_size, grid_size)
                + 1j * interp_imag(pts).reshape(grid_size, grid_size)
            )
           
            # shifted flat-field spectrum (O0=1)
            O0_shift = (
                interp0_real(pts).reshape(grid_size, grid_size)
                + 1j * interp0_imag(pts).reshape(grid_size, grid_size)
            )
            # image fields
            U_obj = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(P * O_shift)))
            U_0   = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(P * O0_shift)))

            I_obj = np.abs(U_obj)**2
            I_0   = np.abs(U_0)**2

            # ===== flat-field normalization (scalar) =====
            # 用空场强度的均值作为该 (xi,eta) 的通光量标尺
            norm = np.mean(I_0) + eps
            I_sum += w * (I_obj / norm)

        images.append(I_sum)

    # ================= 5.5 save results =================
    save_dir = "results_tif"
    os.makedirs(save_dir, exist_ok=True)

    for z, img in zip(z_list, images):
        fname = os.path.join(save_dir, f"partial_coherent_z_{z:+d}um.tif")
        tiff.imwrite(fname, img.astype(np.float32))

    # ================= 6. plot =================
    fig, axes = plt.subplots(1, len(z_list), figsize=(3*len(z_list), 4))
    vmin, vmax = 0.5, 2.0

    for ax, z, img in zip(axes, z_list, images):
        ax.imshow(img, cmap="gray",
                  extent=[x[0], x[-1], y[0], y[-1]],
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"z = {z} µm")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_partial_coherent_imaging()
