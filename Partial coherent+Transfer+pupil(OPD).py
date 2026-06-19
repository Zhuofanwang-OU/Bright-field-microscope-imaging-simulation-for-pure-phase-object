#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partial coherent bright-field simulation using the original OPD phase object.

This is the no-Mie version of the modified SOS model:

    I(x, y) = sum_lambda S_LED(lambda)
              sum_source S_src(xi, eta)
              | F^-1{ P(fx, fy; lambda, z_eff)
                       T(fx, fy; xi, eta, lambda) } |^2

The object is the original pure phase OPD sphere:

    O(x, y; lambda) = exp(i * 2*pi/lambda * (n_bead - n_medium) * thickness)

xi, eta are normalized pupil coordinates. The illumination frequency is:

    f_src = NA_obj * xi / wavelength

Output is raw intensity. No flat-field normalization is applied.
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


def white_led_wavelength_distribution():
    """
    Five-point white LED approximation.

    A narrow blue Gaussian plus a broad phosphor Gaussian is sampled at five
    wavelengths. Returned weights are normalized to sum to 1.
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
    """Uniform source over a normalized circular condenser pupil."""
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


def build_opd_object_field(
    X: np.ndarray,
    Y: np.ndarray,
    wavelength_um: float,
    bead_radius_um: float,
    n_bead: float,
    n_medium: float,
) -> np.ndarray:
    """Build the original pure phase OPD bead object."""
    R = np.sqrt(X**2 + Y**2)
    thickness = 2.0 * np.sqrt(np.maximum(0.0, bead_radius_um**2 - R**2))
    phase_shift = (2.0 * np.pi / wavelength_um) * (n_bead - n_medium) * thickness
    return np.exp(1j * phase_shift).astype(np.complex128)


def simulate_partial_coherent_opd_imaging(
    save_dir: str = "results_tif_opd_led_sos_raw_grid256_pad2_40x_NA06_NAill009_zstep10",
    grid_size: int = 256,
    pad_factor: int = 2,
    sensor_pixel_size_um: float = 3.45,
    magnification: float = 40.0,
    NA_obj: float = 0.60,
    illumination_NA: float = 0.09,
    source_sigma=None,
    n_src: int = 7,
    bead_radius_um: float = 1.0,
    n_bead: float = 1.59,
    n_medium: float = 1.50,
    z_list=None,
    show_plot: bool = True,
):
    """
    Simulate raw bright-field intensity using the OPD phase object.

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
    print("OPD bead radius: %.4f um" % bead_radius_um)
    print("LED wavelength samples (um) and normalized weights:")
    for wl, wt in zip(wavelengths_um, wavelength_weights):
        print("  %.3f um  %.6f" % (wl, wt))
    print("Source points:", len(source_weights), "sigma:", source_sigma)

    for wavelength_um, wavelength_weight in zip(wavelengths_um, wavelength_weights):
        print("Processing wavelength %.3f um" % wavelength_um)

        object_field = build_opd_object_field(
            X=X,
            Y=Y,
            wavelength_um=wavelength_um,
            bead_radius_um=bead_radius_um,
            n_bead=n_bead,
            n_medium=n_medium,
        )
        object_delta = object_field - 1.0

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

            # Only the local OPD perturbation is FFT-propagated. The uniform
            # background plane wave is propagated analytically, which avoids
            # finite-window leakage for non-integer source frequencies.
            delta_transfer_f = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(incident * object_delta))
            )

            source_f_sq = fx_src**2 + fy_src**2
            total_weight = wavelength_weight * source_weight

            for z in z_list:
                background_field = incident * np.exp(
                    -1j * np.pi * wavelength_um * z * source_f_sq
                )
                delta_image_field = np.fft.fftshift(
                    np.fft.ifft2(
                        np.fft.ifftshift(pupil_by_z[z] * delta_transfer_f)
                    )
                )
                image_field = background_field + delta_image_field
                images_pad[z] += total_weight * np.abs(image_field) ** 2

    images = [crop_center(images_pad[z], grid_size) for z in z_list]

    os.makedirs(save_dir, exist_ok=True)
    for z, image in zip(z_list, images):
        fname = os.path.join(save_dir, f"opd_led_sos_raw_z_{z:+d}um.tif")
        tiff.imwrite(fname, image.astype(np.float32))

    metadata_path = os.path.join(save_dir, "simulation_metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("Partial coherent OPD SOS simulation\n")
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
        f.write("bead_radius_um = %.6f\n" % bead_radius_um)
        f.write("bead_diameter_um = %.6f\n" % (2.0 * bead_radius_um))
        f.write("n_bead = %.6f\n" % n_bead)
        f.write("n_medium = %.6f\n" % n_medium)
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
    simulate_partial_coherent_opd_imaging()
