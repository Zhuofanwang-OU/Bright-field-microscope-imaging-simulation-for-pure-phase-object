#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

    # Riccati–Bessel 函数 psi, xi
    psi_x = x * jn_x
    psi_x_der = jn_x + x * jn_x_der
    psi_mx = mx * jn_mx
    psi_mx_der = jn_mx + mx * jn_mx_der

    h_x = jn_x + 1j * yn_x
    h_x_der = jn_x_der + 1j * yn_x_der
    xi_x = x * h_x
    xi_x_der = h_x + x * h_x_der

    # Mie 系数
    a_n = (m * psi_mx * psi_x_der - psi_x * psi_mx_der) / \
          (m * psi_mx * xi_x_der - xi_x * psi_mx_der)

    b_n = (psi_mx * psi_x_der - m * psi_x * psi_mx_der) / \
          (psi_mx * xi_x_der - m * xi_x * psi_mx_der)

    return a_n, b_n


def mie_S1_S2(
    lambda_nm: float = 550.0,
    n_particle: float = 1.59,
    n_medium: float = 1.50,
    diameter_um: float = 2.0,
    num_thetas: int = 721,
):
    """
    计算 λ 固定时的角散射振幅 S1(θ), S2(θ) 及非偏振强度 I(θ)。

    返回:
        theta_deg : 角度 (0–180°)
        I_unpol   : 非偏振强度 (未归一化)
        S1, S2    : 复振幅
    """
    # 半径
    a = (diameter_um * 1e-6) / 2.0
    m = n_particle / n_medium
    lam = lambda_nm * 1e-9

    # size parameter
    x = 2 * np.pi * n_medium * a / lam

    # 级数截断
    nmax = int(round(2 + x + 4 * x**(1/3)))
    n = np.arange(1, nmax + 1, dtype=float)

    a_n, b_n = mie_coefficients(m, x, nmax)

    # 角度采样
    theta_deg = np.linspace(0.0, 180.0, num_thetas)
    theta = np.deg2rad(theta_deg)
    mu = np.cos(theta)

    # 角函数 π_n(mu), τ_n(mu)
    # 递推: π_0 = 0, π_1 = 1
    pi_n = np.zeros((nmax + 1, num_thetas), dtype=float)
    tau_n = np.zeros((nmax + 1, num_thetas), dtype=float)

    pi_n[1, :] = 1.0  # n=1

    for k in range(2, nmax + 1):
        # k 对应阶数 n = k
        # π_k = ((2k-1)/(k-1)) μ π_{k-1} - (k/(k-1)) π_{k-2}
        pi_n[k, :] = ((2 * k - 1) / (k - 1)) * mu * pi_n[k - 1, :] - \
                     (k / (k - 1)) * pi_n[k - 2, :]

    # τ_n = n μ π_n - (n+1) π_{n-1}
    for k in range(1, nmax + 1):
        tau_n[k, :] = k * mu * pi_n[k, :] - (k + 1) * pi_n[k - 1, :]

    # 计算 S1, S2
    S1 = np.zeros(num_thetas, dtype=complex)
    S2 = np.zeros(num_thetas, dtype=complex)

    for k in range(1, nmax + 1):
        factor = (2 * k + 1) / (k * (k + 1))
        an = a_n[k - 1]
        bn = b_n[k - 1]
        S1 += factor * (an * pi_n[k, :] + bn * tau_n[k, :])
        S2 += factor * (an * tau_n[k, :] + bn * pi_n[k, :])

    # 非偏振强度
    I_unpol = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2)

    return theta_deg, I_unpol, S1, S2


if __name__ == "__main__":
    # 你的参数：λ = 550 nm, n_p = 1.59, n_m = 1.50, 直径 2 µm
    theta_deg, I_unpol, S1, S2 = mie_S1_S2(
        lambda_nm=550.0,
        n_particle=1.59,
        n_medium=1.50,
        diameter_um=2.0,
        num_thetas=721,
    )

    # 为了看形状，把强度归一化到 max = 1
    I_norm = I_unpol / np.max(I_unpol)

    # 线性坐标：I(θ)
    plt.figure()
    plt.plot(theta_deg, I_norm)
    plt.xlabel(r"$\theta$ (degrees)")
    plt.ylabel(r"Normalized intensity $I(\theta)$")
    plt.title("Mie angular scattering (λ = 550 nm, d = 2 µm, n=1.59/1.50)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 如果你想看极坐标图，也可以解开下面注释：
    # import matplotlib.pyplot as plt
    # theta_rad = np.deg2rad(theta_deg)
    # plt.figure()
    # ax = plt.subplot(111, projection='polar')
    # ax.plot(theta_rad, I_norm)
    # ax.set_title("Polar plot of I(θ)")
    # plt.show()