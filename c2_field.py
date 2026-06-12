"""
C2 - Tiempo de inversion con campo constante h != 0 (escape metaestable).

Partiendo del estado ordenado m = -1 con un campo h > 0 opuesto, el campo
inclina el doble pozo y rebaja la barrera del lado metaestable. Al crecer h
hacia el campo espinodal h_sp(T) (donde el minimo metaestable desaparece),
<tau> cae de forma abrupta. Es el escape de Kramers/espinodal de campo medio.

  (a) <tau> vs h a T fija, comparando tres modelos de igual Tc = J_medio = 1:
        Curie-Weiss, bloque J12=0.8 (base), bloque J12=0.4.
  (b) <tau> vs h para varias T (bloque base), con h_sp(T) marcado.

Cada temperatura usa su propia ventana medible de h (entre el umbral de
censura y h_sp); definicion de tau: primer paso, inf{t : m(t) >= 0 | m(0)=-1}.
"""

import numpy as np
import matplotlib.pyplot as plt
from block_ising import (BlockIsingParams, BlockIsingModel,
                         constant_field, reversal_time_ensemble)


def linrange(start, stop, n):
    """n valores equiespaciados en [start, stop] (extremos incluidos)."""
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def block_params(N, T, J11=1.2, J12=0.8, p=0.5):
    return BlockIsingParams(N=N, n_blocks=2, proportions=[p, 1 - p],
                            J_intra=J11, J_inter=J12, T=T)


def cw_params(N, T):
    return BlockIsingParams(N=N, n_blocks=1, proportions=[1.0], J_intra=1.0, T=T)


CONFIGS = {
    'Curie-Weiss':            (cw_params,                                   's--'),
    'bloque $J_{12}{=}0.8$':  (lambda N, T: block_params(N, T, 1.2, 0.8),  'o-'),
    'bloque $J_{12}{=}0.4$':  (lambda N, T: block_params(N, T, 1.6, 0.4),  '^-'),
}


def spinodal_field(T, Jbar=1.0):
    """h_sp(T): campo al que desaparece el minimo metaestable en la rama
    simetrica m = tanh(beta(Jbar*m + h)). Comun a los tres modelos (mismo Jbar)."""
    beta = 1.0 / T

    def metastable(h):
        m = -0.95
        for _ in range(5000):
            mn = np.tanh(beta * (Jbar * m + h))
            if abs(mn - m) < 1e-12:
                break
            m = 0.5 * mn + 0.5 * m
        return m < -1e-3

    lo, hi = 0.0, 2.0 * Jbar
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if metastable(mid) else (lo, mid)
    return lo


def mean_tau(params, h, tau_max, n_sim, seed=1):
    model = BlockIsingModel(params, seed=seed)
    r = reversal_time_ensemble(model, constant_field(h), tau_max, n_sim)
    return r['mean'], r['se']


# --------------------------------------------------------------------------
#  Barridos
# --------------------------------------------------------------------------

def sweep_h_configs(h_values, N, T_frac, tau_max, n_sim):
    """<tau>(h) para cada modelo de CONFIGS a una temperatura comun."""
    T = T_frac * 1.0
    res = {lab: ([], []) for lab in CONFIGS}
    for h in h_values:
        for lab, (fac, _) in CONFIGS.items():
            m, se = mean_tau(fac(N, T), h, tau_max, n_sim)
            res[lab][0].append(m); res[lab][1].append(se)
    return np.array(h_values), {l: (np.array(a), np.array(b)) for l, (a, b) in res.items()}


def sweep_h_temps(h_ranges, N, tau_max, n_sim):
    """<tau>(h) del bloque base; h_ranges = {T_frac: (inicio, fin, n_puntos)}."""
    res = {}
    for Tf, rng in h_ranges.items():
        hs = linrange(*rng)
        means, ses = [], []
        for h in hs:
            m, se = mean_tau(block_params(N, Tf * 1.0), h, tau_max, n_sim)
            means.append(m); ses.append(se)
        res[Tf] = (np.array(hs), np.array(means), np.array(ses))
    return res


# --------------------------------------------------------------------------
#  Figura
# --------------------------------------------------------------------------

def make_figure(hc, dC, T_a, dT, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    a = ax[0]
    for lab, (fac, sty) in CONFIGS.items():
        y, e = dC[lab]
        a.errorbar(hc, y, yerr=e, fmt=sty, ms=4, capsize=2, label=lab)
    a.axvline(spinodal_field(T_a), color='gray', ls=':', lw=1, label=r'$h_{sp}$')
    a.set_yscale('log'); a.set_xlabel('h'); a.set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    a.set_title(f'(a) escape vs h   ($T={T_a:.2f}\\,T_c$)'); a.legend(fontsize=8)

    b = ax[1]
    for i, (Tf, (hs, y, e)) in enumerate(sorted(dT.items())):
        col = f'C{i}'
        b.errorbar(hs, y, yerr=e, fmt='o-', ms=4, capsize=2, color=col,
                   label=f'$T={Tf:.2f}\\,T_c$')
        b.axvline(spinodal_field(Tf), color=col, ls=':', lw=1)
    b.set_yscale('log'); b.set_xlabel('h'); b.set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    b.set_title('(b) vs h y T  (punteadas: $h_{sp}$)'); b.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    N_SIM = 1000           # subir en produccion (>= 1000 para ~3% de error)
    TAU_MAX = 40000
    N = 200

    # --- panel (a): un T, ventana de h ---
    T_A = 0.65
    HA_RANGE = (0.08, 0.15, 8)

    # --- panel (b): una ventana de h por temperatura (inicio, fin, nº puntos) ---
    HB_RANGES = {
        0.60: (0.11, 0.18, 8),
        0.65: (0.08, 0.15, 8),
        0.70: (0.05, 0.12, 8),
    }

    hc, dC = sweep_h_configs(linrange(*HA_RANGE), N=N, T_frac=T_A,
                             tau_max=TAU_MAX, n_sim=N_SIM)
    dT = sweep_h_temps(HB_RANGES, N=N, tau_max=TAU_MAX, n_sim=N_SIM)

    make_figure(hc, dC, T_A, dT, 'outputs/c2_field.png')
    print('Figura: c2_field.png')


if __name__ == '__main__':
    main()
