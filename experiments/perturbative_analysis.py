"""
Perturbative regime analysis for reviewer response.

Quantifies how the small-angle approximation sin(ωτ) ≈ ωτ
affects per-mode entropy production estimates, and whether
mode ordering is preserved under the correction.
"""
import numpy as np
import json
from pathlib import Path

# Eigenvalues from Table V (multiasset, seed 42, τ=5 days)
tau = 5.0
modes = [
    # (k, Re, Im)
    (0,  0.400,  0.165),
    (1,  0.400, -0.165),
    (2,  0.425,  0.0),
    (3, -0.210,  0.315),
    (4, -0.210, -0.315),
    (5, -0.311,  0.175),
    (6, -0.311, -0.175),
    (7,  0.031,  0.331),
    (8,  0.031, -0.331),
    (9,  0.183,  0.169),
    (10, 0.183, -0.169),
    (11,-0.158,  0.0),
    (12, 0.095,  0.0),
    (13,-0.013,  0.033),
    (14,-0.013, -0.033),
]

# Per-mode S_k values from Table (appendix)
Sk_perturbative = {
    0: 0.033, 1: 0.022, 2: 0.0,
    3: 0.223, 4: 0.024, 5: 0.145,
    6: 0.173, 7: 0.036, 8: 0.079,
    9: 0.000, 10: 0.031, 11: 0.183,
    12: 0.0, 13: 0.005, 14: 0.010,
}

results = []
print("=" * 85)
print(f"{'Mode':>4} {'|ωτ|':>8} {'sin²(ωτ)':>10} {'(ωτ)²':>10} "
      f"{'Correction':>11} {'Ṡ_pert':>8} {'Ṡ_corr':>8} {'Regime':>12}")
print("-" * 85)

for k, re_val, im_val in modes:
    lam = complex(re_val, im_val)
    mag = abs(lam)
    omega_tau = np.angle(lam)  # arg(λ) = ωτ
    omega = omega_tau / tau
    gamma = -np.log(mag) / tau if mag > 0 else float('inf')

    abs_omega_tau = abs(omega_tau)
    sin2 = np.sin(omega_tau)**2
    omega_tau_sq = omega_tau**2

    # Correction factor: sin²(ωτ) / (ωτ)²
    # Values < 1 mean perturbative overestimates
    if omega_tau_sq > 1e-10:
        correction = sin2 / omega_tau_sq
    else:
        correction = 1.0  # real mode, no correction needed

    # Corrected S_k (multiply perturbative value by correction factor)
    sk_pert = Sk_perturbative[k]
    sk_corr = sk_pert * correction

    # Regime classification
    if abs_omega_tau < 0.3:
        regime = "perturbative"
    elif abs_omega_tau < 1.0:
        regime = "marginal"
    else:
        regime = "NON-PERT"

    results.append({
        'mode': k, 'omega_tau': omega_tau, 'abs_omega_tau': abs_omega_tau,
        'sin2': sin2, 'omega_tau_sq': omega_tau_sq,
        'correction': correction, 'sk_pert': sk_pert, 'sk_corr': sk_corr,
        'regime': regime, 'mag': mag, 'gamma': gamma, 'omega': omega,
    })

    print(f"{k:>4} {abs_omega_tau:>8.3f} {sin2:>10.4f} {omega_tau_sq:>10.4f} "
          f"{correction:>10.3f}x {sk_pert:>8.3f} {sk_corr:>8.3f} {regime:>12}")

print("-" * 85)

# Totals
total_pert = sum(r['sk_pert'] for r in results)
total_corr = sum(r['sk_corr'] for r in results)
knn_estimate = 0.31  # from results

print(f"\nTotal spectral EP (perturbative): {total_pert:.3f} nats/day")
print(f"Total spectral EP (corrected):    {total_corr:.3f} nats/day")
print(f"k-NN estimate (model-free):       {knn_estimate:.3f} nats/day")
print(f"Correction ratio (total):         {total_corr/total_pert:.3f}x")
print(f"Corrected / k-NN ratio:           {total_corr/knn_estimate:.2f}x")

# Check mode ordering preservation
pert_ranking = sorted(results, key=lambda r: r['sk_pert'], reverse=True)
corr_ranking = sorted(results, key=lambda r: r['sk_corr'], reverse=True)

pert_top5 = [r['mode'] for r in pert_ranking[:5]]
corr_top5 = [r['mode'] for r in corr_ranking[:5]]

print(f"\nTop 5 modes (perturbative): {pert_top5}")
print(f"Top 5 modes (corrected):    {corr_top5}")
print(f"Ordering preserved:         {pert_top5 == corr_top5}")

# Count modes in each regime
n_pert = sum(1 for r in results if r['regime'] == 'perturbative')
n_marg = sum(1 for r in results if r['regime'] == 'marginal')
n_nonp = sum(1 for r in results if r['regime'] == 'NON-PERT')
print(f"\nRegime breakdown: {n_pert} perturbative, {n_marg} marginal, {n_nonp} non-perturbative")

# Fraction of EP from non-perturbative modes
ep_nonpert = sum(r['sk_pert'] for r in results if r['regime'] == 'NON-PERT')
print(f"Fraction of perturbative EP from non-perturbative modes: {ep_nonpert/total_pert:.1%}")

# Now do the same for univariate (5 modes)
# Univariate eigenvalues (from paper, approximate from spectral gap = 0.196,
# 2 complex modes out of 5)
# We'll use the reported values: spectral EP = 0.94, k-NN = 0.26
# K_A/K_S ratio = 0.56 (much better)
print("\n" + "=" * 85)
print("UNIVARIATE SUMMARY")
print(f"‖K_A‖/‖K_S‖ ratio: 0.56 (closer to perturbative regime)")
print(f"Spectral EP: 0.94, k-NN: 0.26, ratio: {0.94/0.26:.1f}x")

# Save results
output = {
    'multiasset': {
        'total_perturbative': total_pert,
        'total_corrected': total_corr,
        'knn_reference': knn_estimate,
        'correction_ratio': total_corr / total_pert,
        'corrected_to_knn_ratio': total_corr / knn_estimate,
        'top5_perturbative': pert_top5,
        'top5_corrected': corr_top5,
        'ordering_preserved': pert_top5 == corr_top5,
        'n_perturbative_modes': n_pert,
        'n_marginal_modes': n_marg,
        'n_nonperturbative_modes': n_nonp,
        'fraction_ep_from_nonpert': ep_nonpert / total_pert,
        'modes': [{
            'k': r['mode'], 'omega_tau': r['abs_omega_tau'],
            'correction_factor': r['correction'],
            'sk_perturbative': r['sk_pert'], 'sk_corrected': r['sk_corr'],
            'regime': r['regime']
        } for r in results]
    },
    'univariate': {
        'ka_ks_ratio': 0.56,
        'spectral_ep': 0.94,
        'knn_estimate': 0.26,
    }
}

out_path = Path(__file__).resolve().parent.parent / 'outputs' / 'results' / 'perturbative_analysis.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
