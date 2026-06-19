#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partial coherent bright-field simulation with Mie scattering.

This script follows the modified SOS model used in the PPT:

    I(x, y) = sum_lambda S_LED(lambda)
              sum_source S_src(xi, eta)
              | F^-1{ P(fx, fy; lambda, z_eff)
                       T(fx, fy; xi, eta, lambda) } |^2

where T is generated from the incident tilted plane wave multiplied by a
scalar Mie scattering object field. The source coordinates xi, eta are
normalized pupil coordinates, so the physical frequency shift is
NA_obj * xi / wavelength.

Notes:
- The Mie field uses the complex scalar amplitude (S1 + S2) / 2.
- Absolute scattering strength is not fixed by this scalar model, so
  scattering_scale is an adjustable calibration factor.
- Output is raw intensity; no flat-field normalization is applied.
"""

import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(__file__), ".mplconfig"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy.special import spherical_jn, spherical_yn


def mie_coefficients(m_rel: float, x: float, nmax: int):
    """Compute Mie coefficients a_n and b_n."""
    n = np.arange(1, nmax + 1, dtype=float)

    jn_x = spherical_jn(n, x)
    jn_x_der = spherical_jn(n, x, derivative=True)
    yn_x = spherical_yn(n, x)
    yn_x_der = spherical_yn(n, x, derivative=True)

    mx = m_rel * x
    jn_mx = spherical_jn(n, mx)
    jn_mx_der = spherical_jn(n, mx, derivative=True)

    psi_x = x * jn_x
    psi_x_der = jn_x + x * jn_x_der
    psi_mx = mx * jn_mx
    psi_mx_der = jn_mx + mx * jn_mx_der

    h_x = jn_x + 1j * yn_x
    h_x_der = jn_x_der + 1j * yn_x_der
    xi_x = x * h_x
    xi_x_der = h_x + x * h_x_der

    a_n = (m_rel * psi_mx * psi_x_der - psi_x * psi_mx_der) / (
        m_rel * psi_mx * xi_x_der - xi_x * psi_mx_der
    )
    b_n = (psi_mx * psi_x_der - m_rel * psi_x * psi_mx_der) / (
        psi_mx * xi_x_der - m_rel * xi_x * psi_mx_der
    )
    return a_n, b_n


def mie_s1_s2(
    wavelength_um: float,
    n_particle: float,
    n_medium: float,
    diameter_um: float,
    num_thetas: int = 1441,
):
    """Return theta and complex Mie amplitudes S1, S2."""
    radius_m = diameter_um * 0.5e-6
    wavelength_m = wavelength_um * 1e-6
    m_rel = n_particle / n_medium
    x = 2.0 * np.pi * n_medium * radius_m / wavelength_m
    nmax = int(round(2.0 + x + 4.0 * x ** (1.0 / 3.0)))

    a_n, b_n = mie_coefficients(m_rel, x, nmax)

    theta = np.linspace(0.0, np.pi, num_thetas)
    mu = np.cos(theta)

    pi_n = np.zeros((nmax + 1, num_thetas), dtype=float)
    tau_n = np.zeros((nmax + 1, num_thetas), dtype=float)
    pi_n[1, :] = 1.0

    for order in range(2, nmax + 1):
        pi_n[order, :] = (
            ((2 * order - 1) / (order - 1)) * mu * pi_n[order - 1, :]
            - (order / (order - 1)) * pi_n[order - 2, :]
        )

    for order in range(1, nmax + 1):
        tau_n[order, :] = (
            order * mu * pi_n[order, :] - (order + 1) * pi_n[order - 1, :]
        )

    S1 = np.zeros(num_thetas, dtype=np.complex128)
    S2 = np.zeros(num_thetas, dtype=np.complex128)

    for order in range(1, nmax + 1):
        factor = (2 * order + 1) / (order * (order + 1))
        an = a_n[order - 1]
        bn = b_n[order - 1]
        S1 += factor * (an * pi_n[order, :] + bn * tau_n[order, :])
        S2 += factor * (an * tau_n[order, :] + bn * pi_n[order, :])

    return theta, S1, S2


def mie_scattering_spectrum(
    FX: np.ndarray,
    FY: np.ndarray,
    wavelength_um: float,
    n_particle: float,
    n_medium: float,
    diameter_um: float,
) -> np.ndarray:
    """
    Map complex Mie angular amplitude to a 2D angular spectrum.

    Spatial frequency in the sample medium:
        f_r = n_medium * sin(theta) / wavelength_vacuum
    """
    theta, S1, S2 = mie_s1_s2(
        wavelength_um=wavelength_um,
        n_particle=n_particle,
        n_medium=n_medium,
        diameter_um=diameter_um,
    )
    S_scalar = 0.5 * (S1 + S2)

    F_r = np.sqrt(FX**2 + FY**2)
    sin_theta = wavelength_um * F_r / n_medium
    valid = sin_theta <= 1.0
    theta_map = np.zeros_like(F_r)
    theta_map[valid] = np.arcsin(sin_theta[valid])

    S_real = np.interp(theta_map.ravel(), theta, S_scalar.real, left=0.0, right=0.0)
    S_imag = np.interp(theta_map.ravel(), theta, S_scalar.imag, left=0.0, right=0.0)
    S_map = (S_real + 1j * S_imag).reshape(F_r.shape)
    S_map *= valid

    max_amp = np.max(np.abs(S_map))
    if max_amp > 0:
        S_map /= max_amp
    return S_map


def build_mie_object_field(
    FX: np.ndarray,
    FY: np.ndarray,
    wavelength_um: float,
    n_particle: float,
    n_medium: float,
    diameter_um: float,
    scattering_scale: float,
) -> np.ndarray:
    """
    Build O(x, y; lambda) = 1 + scattering_scale * E_scat(x, y).

    The Mie angular spectrum supplies the complex phase structure; the field is
    peak-normalized because the scalar model leaves the absolute coupling
    coefficient unspecified.
    """
    scatter_f = mie_scattering_spectrum(
        FX=FX,
        FY=FY,
        wavelength_um=wavelength_um,
        n_particle=n_particle,
        n_medium=n_medium,
        diameter_um=diameter_um,
    )
    scatter_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(scatter_f)))

    max_amp = np.max(np.abs(scatter_field))
    if max_amp > 0:
        scatter_field /= max_amp

    return 1.0 + scattering_scale * scatter_field


def white_led_wavelength_distribution():
    """
    Five-point white LED approximation.

    A narrow blue Gaussian plus a broad phosphor Gaussian is sampled at five
    wavelengths. The returned weights are normalized to sum to 1.
    """
    wavelengths_um = np.array([0.450, 0.500, 0.550, 0.600, 0.650], dtype=float)

    blue_center_um = 0.455
    blue_fwhm_um = 0.020
    blue_amp = 1.00

    phosphor_center_um = 0.570
    phosphor_fwhm_um = 0.130
    phosphor_amp = 0.85

    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    blue_sigma = blue_fwhm_um * fwhm_to_sigma
    phosphor_sigma = phosphor_fwhm_um * fwhm_to_sigma

    blue = blue_amp * np.exp(
        -0.5 * ((wavelengths_um - blue_center_um) / blue_sigma) ** 2
    )
    phosphor = phosphor_amp * np.exp(
        -0.5 * ((wavelengths_um - phosphor_center_um) / phosphor_sigma) ** 2
    )
    weights = blue + phosphor
    weights /= np.sum(weights)
    return wavelengths_um, weights


def normalized_source_distribution(sigma: float, n_src: int):
    """
    Uniform source over a normalized circular condenser pupil.

    xi, eta are normalized to the objective pupil radius. The actual
    illumination frequency is:
        f_src = NA_obj * xi / wavelength
    """
    src_lin = np.linspace(-sigma, sigma, n_src)
    xi_grid, eta_grid = np.meshgrid(src_lin, src_lin)
    src_mask = xi_grid**2 + eta_grid**2 <= sigma**2

    xi_list = xi_grid[src_mask]
    eta_list = eta_grid[src_mask]
    weights = np.ones_like(xi_list, dtype=float)
    weights /= np.sum(weights)
    return xi_list, eta_list, weights


def crop_center(arr: np.ndarray, target_size: int):
    if arr.shape[0] == target_size:
        return arr
    start = (arr.shape[0] - target_size) // 2
    return arr[start : start + target_size, start : start + target_size]


def simulate_partial_coherent_mie_imaging(
    save_dir: str = "results_tif_mie_led_sos_raw_grid256_pad2_40x_NA06_NAill009_zstep10",
    grid_size: int = 256,
    pad_factor: int = 2,
    sensor_pixel_size_um: float = 3.45,
    magnification: float = 40.0,
    NA_obj: float = 0.60,
    illumination_NA: float = 0.09,
    source_sigma=None,
    n_src: int = 7,
    bead_diameter_um: float = 2.0,
    n_bead: float = 1.59,
    n_medium: float = 1.50,
    scattering_scale: float = 0.18,
    z_list=None,
    show_plot: bool = True,
):
    """
    Simulate raw bright-field intensity for a partially coherent LED microscope.

    +z is interpreted as the sample moving away from the objective, and z_eff=z.
    The pupil phase follows the PPT sign:
        P = A * exp(-i*pi*lambda*z_eff*(fx^2 + fy^2))
    """
    if z_list is None:
        z_list = list(range(-50, 51, 10))

    pixel_size = sensor_pixel_size_um / magnification
    if source_sigma is None:
        source_sigma = illumination_NA / NA_obj

    n_pad = int(grid_size * pad_factor)
    if n_pad % 2 != 0:
        n_pad += 1

    x = (np.arange(n_pad) - n_pad / 2) * pixel_size
    y = (np.arange(n_pad) - n_pad / 2) * pixel_size
    X, Y = np.meshgrid(x, y)

    fx = np.fft.fftshift(np.fft.fftfreq(n_pad, d=pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(n_pad, d=pixel_size))
    FX, FY = np.meshgrid(fx, fy)
    F_sq = FX**2 + FY**2

    wavelengths_um, wavelength_weights = white_led_wavelength_distribution()
    xi_list, eta_list, source_weights = normalized_source_distribution(
        sigma=source_sigma,
        n_src=n_src,
    )

    images_pad = {z: np.zeros((n_pad, n_pad), dtype=float) for z in z_list}

    print("Object-plane pixel size: %.4f um/pixel" % pixel_size)
    print("LED wavelength samples (um) and normalized weights:")
    for wl, wt in zip(wavelengths_um, wavelength_weights):
        print("  %.3f um  %.6f" % (wl, wt))
    print("Source points:", len(source_weights), "sigma:", source_sigma)

    for wavelength_um, wavelength_weight in zip(wavelengths_um, wavelength_weights):
        print("Processing wavelength %.3f um" % wavelength_um)

        object_field = build_mie_object_field(
            FX=FX,
            FY=FY,
            wavelength_um=wavelength_um,
            n_particle=n_bead,
            n_medium=n_medium,
            diameter_um=bead_diameter_um,
            scattering_scale=scattering_scale,
        )

        f_cutoff = NA_obj / wavelength_um
        pupil_amp = (F_sq <= f_cutoff**2).astype(float)

        pupil_by_z = {
            z: pupil_amp * np.exp(-1j * np.pi * wavelength_um * z * F_sq)
            for z in z_list
        }

        for xi_norm, eta_norm, source_weight in zip(
            xi_list, eta_list, source_weights
        ):
            fx_src = NA_obj * xi_norm / wavelength_um
            fy_src = NA_obj * eta_norm / wavelength_um
            incident = np.exp(1j * 2.0 * np.pi * (fx_src * X + fy_src * Y))

            transfer_f = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(incident * object_field))
            )

            total_weight = wavelength_weight * source_weight
            for z in z_list:
                image_field = np.fft.fftshift(
                    np.fft.ifft2(np.fft.ifftshift(pupil_by_z[z] * transfer_f))
                )
                images_pad[z] += total_weight * np.abs(image_field) ** 2

    images = [crop_center(images_pad[z], grid_size) for z in z_list]

    os.makedirs(save_dir, exist_ok=True)
    for z, image in zip(z_list, images):
        fname = os.path.join(save_dir, f"mie_led_sos_raw_z_{z:+d}um.tif")
        tiff.imwrite(fname, image.astype(np.float32))

    metadata_path = os.path.join(save_dir, "simulation_metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("Partial coherent Mie SOS simulation\n")
        f.write("raw intensity output, no flat-field normalization\n")
        f.write("sensor_pixel_size_um = %.6f\n" % sensor_pixel_size_um)
        f.write("magnification = %.6f\n" % magnification)
        f.write("object_pixel_size_um = %.6f\n" % pixel_size)
        f.write("NA_obj = %.6f\n" % NA_obj)
        f.write("illumination_NA = %.6f\n" % illumination_NA)
        f.write("source_sigma = %.6f\n" % source_sigma)
        f.write("effective_NA_ill = %.6f\n" % (source_sigma * NA_obj))
        f.write("n_src = %d\n" % n_src)
        f.write("source_points = %d\n" % len(source_weights))
        f.write("bead_diameter_um = %.6f\n" % bead_diameter_um)
        f.write("n_bead = %.6f\n" % n_bead)
        f.write("n_medium = %.6f\n" % n_medium)
        f.write("scattering_scale = %.6f\n" % scattering_scale)
        f.write("z_list_um = %s\n" % list(z_list))
        f.write("wavelength_um,normalized_weight\n")
        for wl, wt in zip(wavelengths_um, wavelength_weights):
            f.write("%.6f,%.12f\n" % (wl, wt))

    if show_plot:
        x_crop = (np.arange(grid_size) - grid_size / 2) * pixel_size
        y_crop = (np.arange(grid_size) - grid_size / 2) * pixel_size
        vmin = min(float(np.min(image)) for image in images)
        vmax = max(float(np.max(image)) for image in images)

        fig, axes = plt.subplots(1, len(z_list), figsize=(3 * len(z_list), 4))
        if len(z_list) == 1:
            axes = [axes]
        for ax, z, image in zip(axes, z_list, images):
            ax.imshow(
                image,
                cmap="gray",
                extent=[x_crop[0], x_crop[-1], y_crop[0], y_crop[-1]],
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"z = {z} um")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    return images


if __name__ == "__main__":
    simulate_partial_coherent_mie_imaging()
