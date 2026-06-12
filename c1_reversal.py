"""
C1 - Tiempo de inversion con campo nulo (h = 0).

Mide <tau> (definicion de primer paso: tau = inf{t : m(t) >= 0 | m(0) = -1}):
  (a) <tau> vs T  : Tres modelos de igual Tc = J_medio = 1
                    (Curie-Weiss, bloque J12=0.8, bloque J12=0.4).
  (b) <tau> vs N  : barrera ~ e^{cN}. Mismos tres modelos.
  (c) <tau> vs J12: firma de bloques a Tc = J_medio = 1 (J11=J22=2-J12), para
                    dos temperaturas: el efecto se intensifica al bajar T/Tc.
  (d) <tau> vs p  : proporcion de bloques, para acoplamiento fuerte y debil;
                    p solo influye cuando los bloques se desacoplan.

Nota (d): variar p con acoplamientos fijos co-varia Tc, J_medio y el tamano de
los bloques; se mantiene T/Tc fijo, pero la curva mezcla esos efectos.
"""

import numpy as np
import matplotlib.pyplot as plt
from block_ising import (BlockIsingParams, BlockIsingModel,
                         constant_field, reversal_time_ensemble)


def linrange(start, stop, n):
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def block_params(N, T, J11=1.2, J12=0.8, p=0.5):
    return BlockIsingParams(N=N, n_blocks=2, proportions=[p, 1 - p],
                            J_intra=J11, J_inter=J12, T=T)


def cw_params(N, T):
    return BlockIsingParams(N=N, n_blocks=1, proportions=[1.0], J_intra=1.0, T=T)


# Modelos comparados en (a) y (b): label -> (factory(N,T), estilo)
CONFIGS = {
    'Curie-Weiss':            (cw_params,                                   's--'),
    'bloque $J_{12}{=}0.8$':  (lambda N, T: block_params(N, T, 1.2, 0.8),  'o-'),
    'bloque $J_{12}{=}0.4$':  (lambda N, T: block_params(N, T, 1.6, 0.4),  '^-'),
}

# Acoplamientos comparados en (d): label -> (J11, J12)
COUPLINGS_P = {
    'fuerte $J_{12}{=}0.8$': (1.2, 0.8),
    'debil $J_{12}{=}0.1$': (1.9, 0.1),
}


def mean_tau(params, tau_max, n_sim, seed=1):
    model = BlockIsingModel(params, seed=seed)
    r = reversal_time_ensemble(model, constant_field(0.0), tau_max, n_sim)
    return r['mean'], r['se']


# --------------------------------------------------------------------------
#  Barridos
# --------------------------------------------------------------------------

def sweep_T(T_fracs, N, tau_max, n_sim):
    res = {lab: ([], []) for lab in CONFIGS}
    for Tf in T_fracs:
        for lab, (fac, _) in CONFIGS.items():
            m, se = mean_tau(fac(N, Tf * 1.0), tau_max, n_sim)
            res[lab][0].append(m); res[lab][1].append(se)
    return np.array(T_fracs), {l: (np.array(a), np.array(b)) for l, (a, b) in res.items()}


def sweep_N(N_values, T_frac, tau_max, n_sim):
    res = {lab: ([], []) for lab in CONFIGS}
    for N in N_values:
        N = int(round(N))
        for lab, (fac, _) in CONFIGS.items():
            m, se = mean_tau(fac(N, T_frac), tau_max, n_sim)
            res[lab][0].append(m); res[lab][1].append(se)
    return np.array(N_values), {l: (np.array(a), np.array(b)) for l, (a, b) in res.items()}


def sweep_J12(J12_values, N, T_fracs, tau_max, n_sim):
    """<tau> vs J12 con J11=J22=2-J12 (Tc=J_medio=1), para varias T/Tc."""
    out = {}
    for Tf in T_fracs:
        tau, se = [], []
        for J12 in J12_values:
            m, s = mean_tau(block_params(N, Tf * 1.0, J11=2 - J12, J12=J12), tau_max, n_sim)
            tau.append(m); se.append(s)
        out[Tf] = (np.array(J12_values), np.array(tau), np.array(se))
    return out


def sweep_p(p_values, N, T_frac, tau_max, n_sim):
    """<tau> vs p para cada acoplamiento de COUPLINGS_P (Tc recalculado por p)."""
    out = {}
    for lab, (J11, J12) in COUPLINGS_P.items():
        tau, se = [], []
        for p in p_values:
            Tc = block_params(N, 1.0, J11, J12, p).Tc
            m, s = mean_tau(block_params(N, T_frac * Tc, J11, J12, p), tau_max, n_sim)
            tau.append(m); se.append(s)
        out[lab] = (np.array(p_values), np.array(tau), np.array(se))
    return out


# --------------------------------------------------------------------------
#  Figura
# --------------------------------------------------------------------------

def make_figure(Tf, dT, Nv, dN, dJ, dp, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    for lab, (fac, sty) in CONFIGS.items():
        a.errorbar(1 / Tf, dT[lab][0], yerr=dT[lab][1], fmt=sty, ms=4, capsize=2, label=lab)
    a.set_yscale('log'); a.set_xlabel(r'$T_c/T$'); a.set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    a.set_title('(a) $\\langle\\tau\\rangle$ vs T '); a.legend(fontsize=8)

    b = ax[0, 1]
    for lab, (fac, sty) in CONFIGS.items():
        b.errorbar(Nv, dN[lab][0], yerr=dN[lab][1], fmt=sty, ms=4, capsize=2, label=lab)
    b.set_yscale('log'); b.set_xlabel('N'); b.set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    b.set_title('(b) $\\langle\\tau\\rangle$ vs N   [barrera $\\sim e^{cN}$]'); b.legend(fontsize=8)

    c = ax[1, 0]
    for i, (Tf_c, (J12, tau, se)) in enumerate(sorted(dJ.items())):
        c.errorbar(J12, tau, yerr=se, fmt='o-', ms=4, capsize=2, color=f'C{i}',
                   label=f'$T={Tf_c:.2f}\\,T_c$')
    c.set_yscale('log'); c.set_xlabel(r'$J_{12}$'); c.set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    c.set_title('(c) $\\langle\\tau\\rangle$ vs $J_{12}$  ($T_c,\\bar J$ fijos)'); c.legend(fontsize=8)

    d = ax[1, 1]
    for i, (lab, (pv, tau, se)) in enumerate(dp.items()):
        d.errorbar(pv, tau, yerr=se, fmt='o-', ms=4, capsize=2, color=f'C{i+2}', label=lab)
    d.set_yscale('log'); d.set_xlabel(r'$p$ (bloque 1)'); d.set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    d.set_title('(d) $\\langle\\tau\\rangle$ vs $p$'); d.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    N_SIM = 1000           # subir en produccion (>= 1000 para ~3% de error)
    TAU_MAX = 100000

    # --- rangos ajustables: (inicio, fin, nº de puntos) ---
    T_RANGE   = (0.84, 0.98, 8)
    N_RANGE   = (50, 350, 8)
    J12_RANGE = (0.1, 0.9, 9)
    P_RANGE   = (0.15, 0.50, 8)
    T_J12 = [0.85, 0.87, 0.90]     # temperaturas del panel (c)
    T_P = 0.90               # temperatura del panel (d)

    Tf, dT = sweep_T(linrange(*T_RANGE), N=200, tau_max=TAU_MAX, n_sim=N_SIM)
    Nv, dN = sweep_N(linrange(*N_RANGE), T_frac=0.88, tau_max=TAU_MAX, n_sim=N_SIM)
    dJ = sweep_J12(linrange(*J12_RANGE), N=200, T_fracs=T_J12,
                   tau_max=TAU_MAX, n_sim=N_SIM)
    dp = sweep_p(linrange(*P_RANGE), N=200, T_frac=T_P,
                 tau_max=TAU_MAX, n_sim=N_SIM)

    make_figure(Tf, dT, Nv, dN, dJ, dp, 'outputs/c1_reversal.png')
    print('Figura: c1_reversal.png')


if __name__ == '__main__':
    main()
