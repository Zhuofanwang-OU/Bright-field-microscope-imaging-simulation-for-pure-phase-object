#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:08:33 2025

@author: wangzhuofan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn


def mie_coefficients(m: complex, x: float, nmax: int):
    """计算 Mie 系数 a_n, b_n."""
    n = np.arange(1, nmax + 1, dtype=float)

    # 在 x 处的球贝塞尔函数
    jn_x = spherical_jn(n, x)
    jn_x_der = spherical_jn(n, x, derivative=True)
    yn_x = spherical_yn(n, x)
    yn_x_der = spherical_yn(n, x, derivative=True)

    # 在 m x 处
    mx = m * x
    jn_mx = spherical_jn(n, mx)
    jn_mx_der = spherical_jn(n, mx, derivative=True)
    yn_mx = spherical_yn(n, mx)
    yn_mx_der = spherical_yn(n, mx, derivative=True)

    # Riccati–Bessel: psi, xi
    psi_x = x * jn_x
    psi_x_der = jn_x + x * jn_x_der
    psi_mx = mx * jn_mx
    psi_mx_der = jn_mx + mx * jn_mx_der

    h_x = jn_x + 1j * yn_x
    h_x_der = jn_x_der + 1j * yn_x_der
    xi_x = x * h_x
    xi_x_der = h_x + x * h_x_der

    a_n = (m * psi_mx * psi_x_der - psi_x * psi_mx_der) / \
          (m * psi_mx * xi_x_der - xi_x * psi_mx_der)
    b_n = (psi_mx * psi_x_der - m * psi_x * psi_mx_der) / \
          (psi_mx * xi_x_der - m * xi_x * psi_mx_der)

    return a_n, b_n


def mie_S1_S2(lambda_nm: float = 550.0,
              n_particle: float = 1.59,
              n_medium: float = 1.50,
              diameter_um: float = 2.0,
              num_thetas: int = 721):
    """
    计算 λ 固定时的 S1(θ), S2(θ) 和非偏振强度 I(θ)。
    """
    # 几何参数
    a = (diameter_um * 1e-6) / 2.0  # 半径 [m]
    m = n_particle / n_medium
    lam = lambda_nm * 1e-9          # 波长 [m]

    # size parameter
    x = 2 * np.pi * n_medium * a / lam

    # 级数截断
    nmax = int(round(2 + x + 4 * x**(1/3)))
    a_n, b_n = mie_coefficients(m, x, nmax)

    # 角度网格
    theta_deg = np.linspace(0.0, 180.0, num_thetas)
    theta = np.deg2rad(theta_deg)
    mu = np.cos(theta)

    # 角函数 π_n, τ_n
    pi_n = np.zeros((nmax + 1, num_thetas), dtype=float)
    tau_n = np.zeros((nmax + 1, num_thetas), dtype=float)

    pi_n[1, :] = 1.0  # n = 1

    for k in range(2, nmax + 1):
        pi_n[k, :] = ((2 * k - 1) / (k - 1)) * mu * pi_n[k - 1, :] - \
                     (k / (k - 1)) * pi_n[k - 2, :]

    for k in range(1, nmax + 1):
        tau_n[k, :] = k * mu * pi_n[k, :] - (k + 1) * pi_n[k - 1, :]

    # S1, S2
    S1 = np.zeros(num_thetas, dtype=complex)
    S2 = np.zeros(num_thetas, dtype=complex)

    n_arr = np.arange(1, nmax + 1, dtype=float)
    for k in range(1, nmax + 1):
        factor = (2 * k + 1) / (k * (k + 1))
        an = a_n[k - 1]
        bn = b_n[k - 1]
        S1 += factor * (an * pi_n[k, :] + bn * tau_n[k, :])
        S2 += factor * (an * tau_n[k, :] + bn * pi_n[k, :])

    I_unpol = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2)

    return theta_deg, theta, I_unpol


if __name__ == "__main__":
    # 计算 λ = 550 nm 的角度分布
    theta_deg, theta_rad, I_unpol = mie_S1_S2(
        lambda_nm=550.0,
        n_particle=1.59,
        n_medium=1.50,
        diameter_um=2.0,
        num_thetas=721,
    )

    # --- 画成类似书上的极坐标对数图 ---
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="polar")

    # 极坐标：0° 在右侧（E），顺时针增大角度 → 和示意图一致
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    # 使用对数半径刻度
    ax.plot(theta_rad, I_unpol)
    ax.set_rscale("log")

    # 设置角度刻度
    ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180])

    ax.set_title("Mie scattering pattern\n"
                 "λ = 550 nm, d = 2 µm, n = 1.59 / 1.50",
                 pad=20)

    plt.tight_layout()
    plt.show()

    # --- 计算“有多少散射集中在前向若干度内” ---
    dtheta = theta_rad[1] - theta_rad[0]

    # 功率权重 ∝ I(θ) sinθ dθ（周向已积分，因此 2π 省略）
    weights = I_unpol * np.sin(theta_rad) * dtheta
    total_power = np.sum(weights)
    cdf = np.cumsum(weights) / total_power

    def angle_for_fraction(frac: float) -> float:
        idx = np.searchsorted(cdf, frac)
        return float(theta_deg[idx])

    angle_50 = angle_for_fraction(0.5)
    angle_90 = angle_for_fraction(0.9)

    print(f"Half-power angle (50% power in 0–θ): {angle_50:.2f} deg")
    print(f"0–θ cone containing 90% of total power: {angle_90:.2f} deg")