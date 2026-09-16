"""
Rossby wave propagation: a pseudo-spectral simulation.

Large-scale atmospheric waves (Rossby waves) exist because the
Coriolis parameter changes with latitude (the "beta effect"). On a
beta-plane with background zonal wind U, linear theory predicts a
wave with wavenumber k propagates at phase speed:

    c = U - beta / k^2

This script numerically solves the (nonlinear) barotropic vorticity
equation with a pseudo-spectral method (FFT-based spatial
derivatives, RK4 time-stepping, 2/3-rule dealiasing) for a single
wave riding on a background wind, and checks that the simulated
phase speed matches the analytical formula above.
"""

import numpy as np
import matplotlib.pyplot as plt

Omega = 7.292e-5
R_earth = 6.371e6
lat0 = 45.0
beta = 2 * Omega * np.cos(np.radians(lat0)) / R_earth

U = 15.0
k_s = np.sqrt(beta / U)
Lx = Ly = 2 * np.pi / k_s

nx = ny = 64
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y, indexing="ij")

kx1 = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
ky1 = np.fft.fftfreq(ny, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx1, ky1, indexing="ij")
K2 = KX**2 + KY**2
K2_safe = K2.copy()
K2_safe[0, 0] = 1.0

kx_max, ky_max = np.max(np.abs(kx1)), np.max(np.abs(ky1))
dealias = (np.abs(KX) < (2/3) * kx_max) & (np.abs(KY) < (2/3) * ky_max)

def rhs(zeta_hat):
    psi_hat = -zeta_hat / K2_safe
    psi_hat[0, 0] = 0.0

    u = np.real(np.fft.ifft2(-1j * KY * psi_hat))
    v = np.real(np.fft.ifft2(1j * KX * psi_hat))
    zx = np.real(np.fft.ifft2(1j * KX * zeta_hat))
    zy = np.real(np.fft.ifft2(1j * KY * zeta_hat))

    nonlinear_hat = np.fft.fft2(u * zx + v * zy) * dealias
    linear_hat = 1j * KX * U * zeta_hat + 1j * KX * beta * psi_hat
    return -nonlinear_hat - linear_hat

def run_simulation(m_wave, n_steps, dt):
    k_m = 2 * np.pi * m_wave / Lx
    v_amp = 0.3
    psi0 = v_amp / k_m
    zeta_hat = np.fft.fft2(-k_m**2 * psi0 * np.cos(k_m * X))

    times = np.zeros(n_steps)
    phases = np.zeros(n_steps)
    hovmoller = np.zeros((n_steps, nx))

    for i in range(n_steps):
        psi_hat = -zeta_hat / K2_safe
        psi_hat[0, 0] = 0.0
        psi_phys = np.real(np.fft.ifft2(psi_hat))
        hovmoller[i, :] = psi_phys[:, 0]

        phases[i] = np.angle(zeta_hat[m_wave, 0])
        times[i] = i * dt

        k1 = rhs(zeta_hat)
        k2 = rhs(zeta_hat + 0.5 * dt * k1)
        k3 = rhs(zeta_hat + 0.5 * dt * k2)
        k4 = rhs(zeta_hat + dt * k3)
        zeta_hat = zeta_hat + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return times, phases, hovmoller, k_m

if __name__ == "__main__":
    dt = 1000.0
    m_wave = 3
    n_steps = 700
    times, phases, hovmoller, k_m = run_simulation(m_wave, n_steps, dt)

    c_analytic = U - beta / k_m**2

    unwrapped = np.unwrap(phases)
    slope = np.polyfit(times, unwrapped, 1)[0]
    c_numeric = -slope / k_m

    print(f"Domain size: {Lx/1000:.0f} km, wavenumber m = {m_wave}")
    print(f"Analytical phase speed: {c_analytic:.4f} m/s")
    print(f"Numerical phase speed:  {c_numeric:.4f} m/s")

    plt.figure(figsize=(8, 6))
    plt.imshow(hovmoller, aspect="auto", origin="lower",
               extent=[0, Lx / 1000, 0, times[-1] / 86400], cmap="RdBu_r")
    plt.colorbar(label="Streamfunction perturbation (m^2/s)")
    plt.xlabel("x (km)")
    plt.ylabel("Time (days)")
    plt.title(f"Hovmoller diagram: Rossby wave (m={m_wave}) propagation")
    plt.savefig("rossby_hovmoller.png")
    print("Saved rossby_hovmoller.png")

    plt.figure(figsize=(8, 5))
    plt.plot(times / 86400, unwrapped, label="Numerical (unwrapped phase)")
    plt.plot(times / 86400, phases[0] - k_m * c_analytic * times,
              '--', label="Analytical prediction")
    plt.xlabel("Time (days)")
    plt.ylabel("Wave phase (radians)")
    plt.title("Rossby wave phase evolution: simulation vs theory")
    plt.legend()
    plt.savefig("rossby_phase_check.png")
    print("Saved rossby_phase_check.png")

    rel_error = abs(c_numeric - c_analytic) / abs(c_analytic)
    print(f"Relative error: {rel_error:.2e}")
    assert rel_error < 0.01, "Simulated phase speed doesn't match Rossby wave theory!"
    print("PASS: pseudo-spectral simulation matches the analytical Rossby wave dispersion relation.")
