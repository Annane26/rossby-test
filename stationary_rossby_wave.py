"""
Stationary Rossby waves and the planetary wave dispersion relation.

Rossby wave phase speed is c = U - beta/k^2. Since this is westward
RELATIVE to the mean flow, there exists a special "stationary"
wavenumber k_s = sqrt(beta/U) where the wave doesn't move at all in
the ground-fixed frame. This is the real mechanism behind
quasi-permanent troughs and ridges in the mid-latitude jet stream
(e.g. the wintertime Aleutian Low / Icelandic Low).

Reuses the pseudo-spectral solver from rossby_wave_hovmoller.py at
several wavenumbers, including exactly the stationary one, and
checks the result against the full dispersion curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from rossby_wave_hovmoller import (
    U, beta, Lx, k_s, run_simulation
)

if __name__ == "__main__":
    dt = 1000.0
    wave_numbers = [1, 2, 3, 4, 5, 6]
    results = []

    for m in wave_numbers:
        k_m = 2 * np.pi * m / Lx
        c_analytic = U - beta / k_m**2
        omega = k_m * c_analytic

        if abs(omega) < 1e-10:
            n_steps = 400
        else:
            period = 2 * np.pi / abs(omega)
            n_steps = int(np.clip(4 * period / dt, 200, 2000))

        times, phases, _, _ = run_simulation(m, n_steps, dt)
        unwrapped = np.unwrap(phases)
        slope = np.polyfit(times, unwrapped, 1)[0]
        c_numeric = -slope / k_m

        results.append((m, k_m, c_analytic, c_numeric))
        print(f"m={m}: k={k_m:.2e} 1/m, analytic c={c_analytic:7.3f} m/s, "
              f"numeric c={c_numeric:7.3f} m/s")

    k_continuous = np.linspace(0.5 * (2 * np.pi / Lx), 6.5 * (2 * np.pi / Lx), 300)
    c_continuous = U - beta / k_continuous**2

    plt.figure(figsize=(8, 6))
    plt.plot(k_continuous * 1e6, c_continuous, label="Analytical: c = U - beta/k^2", color="black")
    ks_sim = [r[1] for r in results]
    cs_sim = [r[3] for r in results]
    plt.scatter(np.array(ks_sim) * 1e6, cs_sim, color="tab:red", zorder=5,
                label="Simulated (pseudo-spectral model)")
    plt.axhline(0, color="gray", linewidth=0.7)
    plt.axvline(k_s * 1e6, color="blue", linestyle=":", label="Stationary wavenumber k_s")
    plt.xlabel("Wavenumber k (x1e-6 rad/m)")
    plt.ylabel("Phase speed c (m/s)")
    plt.title("Rossby wave dispersion relation: theory vs simulation")
    plt.legend()
    plt.savefig("rossby_dispersion_relation.png")
    print("Saved rossby_dispersion_relation.png")

    max_rel_error = 0.0
    for m, k_m, c_a, c_n in results:
        denom = max(abs(c_a), 0.05)
        rel_error = abs(c_n - c_a) / denom
        max_rel_error = max(max_rel_error, rel_error)

    print(f"Max relative/absolute error across all wavenumbers: {max_rel_error:.2e}")
    assert max_rel_error < 0.02, "Simulated dispersion relation doesn't match theory!"

    m1_result = results[0]
    print(f"m=1 (built at k_s) numerical phase speed: {m1_result[3]:.4f} m/s")
    assert abs(m1_result[3]) < 0.05, "The wave at k_s should be nearly stationary!"

    print("PASS: simulated phase speeds match the theoretical Rossby wave dispersion "
          "relation, and the wave at k_s is confirmed stationary.")
