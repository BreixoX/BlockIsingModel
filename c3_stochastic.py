"""
C3 - Tiempo de inversion con campo estocastico gaussiano (Toral et al. 2025).

Campo espacialmente uniforme h(t) ~ N(0, 2D), remuestreado una vez por MCS.
Para que el rango D in [0, 0.6] sea informativo se usa el modelo reescalado a
Tc = 4 (acoplamientos x4), que coincide con la descripcion de campo medio de
Toral (numero de coordinacion 4, Tc^MF = 4; anexos SM3-SM5).

  (a) <tau> vs D a T fija (estilo Naskar: tau decrece con D), tres modelos de
      igual Tc = J_medio = 4: Curie-Weiss, bloque J12=3.2 (base), bloque J12=1.6.
  (b) Diagrama (D,T): fronteras soft-ferro / ferro para una familia de tau_max,
      extraidas de la malla <tau>(D,T) como el contorno <tau> = tau_max
      (analogo a la Fig. SM5 de Toral).

Definicion de tau: primer paso, tau = inf{t : m(t) >= 0 | m(0) = -1}.
"""

import numpy as np
import matplotlib.pyplot as plt
from block_ising import (BlockIsingParams, BlockIsingModel,
                         gaussian_field, reversal_time_ensemble)

TC = 4.0   # temperatura critica del modelo reescalado


def linrange(start, stop, n):
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def block_params(N, T, J11=4.8, J12=3.2, p=0.5):
    return BlockIsingParams(N=N, n_blocks=2, proportions=[p, 1 - p],
                            J_intra=J11, J_inter=J12, T=T)


def cw_params(N, T):
    return BlockIsingParams(N=N, n_blocks=1, proportions=[1.0], J_intra=4.0, T=T)


CONFIGS = {
    'Curie-Weiss':            (cw_params,                                  's--'),
    'bloque $J_{12}{=}3.2$':  (lambda N, T: block_params(N, T, 4.8, 3.2),  'o-'),
    'bloque $J_{12}{=}1.6$':  (lambda N, T: block_params(N, T, 6.4, 1.6),  '^-'),
}


def mean_tau(params, D, tau_max, n_sim, seed=1):
    model = BlockIsingModel(params, seed=seed)
    r = reversal_time_ensemble(model, gaussian_field(D), tau_max, n_sim)
    return r['mean'], r['se']


# --------------------------------------------------------------------------
#  (a) <tau> vs D
# --------------------------------------------------------------------------

def sweep_D_configs(D_values, N, T_frac, tau_max, n_sim):
    T = T_frac * TC
    res = {lab: ([], []) for lab in CONFIGS}
    for D in D_values:
        for lab, (fac, _) in CONFIGS.items():
            m, se = mean_tau(fac(N, T), D, tau_max, n_sim)
            res[lab][0].append(m); res[lab][1].append(se)
    return np.array(D_values), {l: (np.array(a), np.array(b)) for l, (a, b) in res.items()}


# --------------------------------------------------------------------------
#  (b) malla <tau>(D,T) y extraccion de fronteras
# --------------------------------------------------------------------------

def grid_tau(D_grid, T_grid, N, cap, n_sim):
    """<tau> en cada (D,T); inf donde se censura (regimen ferromagnetico)."""
    G = np.full((len(T_grid), len(D_grid)), np.inf)
    for i, T in enumerate(T_grid):
        for j, D in enumerate(D_grid):
            m, _ = mean_tau(block_params(N, T), D, cap, n_sim)
            if np.isfinite(m):
                G[i, j] = m
    return G


def boundary(D_grid, tau_row, tau_max):
    """D donde <tau>(D) cruza tau_max en una fila de T (<tau> decrece con D)."""
    y = np.where(np.isfinite(tau_row), np.log(tau_row), np.inf)
    t = np.log(tau_max)
    for j in range(len(D_grid) - 1):
        if y[j] >= t >= y[j + 1] and np.isfinite(y[j + 1]):
            if np.isinf(y[j]):
                return D_grid[j + 1]               # cruce contra zona censurada
            f = (y[j] - t) / (y[j] - y[j + 1])
            return D_grid[j] + f * (D_grid[j + 1] - D_grid[j])
    return None


# --------------------------------------------------------------------------
#  Figura
# --------------------------------------------------------------------------

def make_figure(Dv, dC, T_a, D_grid, T_grid, G, tau_family, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    a = ax[0]
    for lab, (fac, sty) in CONFIGS.items():
        y, e = dC[lab]
        a.errorbar(Dv, y, yerr=e, fmt=sty, ms=4, capsize=2, label=lab)
    a.set_yscale('log'); a.set_xlabel('D'); a.set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    a.set_title(f'(a) $\\langle\\tau\\rangle$ vs D   ($T={T_a:.2f}\\,T_c$)'); a.legend(fontsize=8)

    b = ax[1]
    for k, tm in enumerate(tau_family):
        Ds, Ts = [], []
        for i, T in enumerate(T_grid):
            Db = boundary(D_grid, G[i], tm)
            if Db is not None:
                Ds.append(Db); Ts.append(T)
        if Ds:
            b.plot(Ds, Ts, 'o-', ms=4, color=f'C{k}',
                   label=r'$\tau_{max}=10^{%d}$' % int(round(np.log10(tm))))
    b.axhline(TC, color='k', ls='--', lw=1, label=r'$T_c^{MF}=4$')
    b.set_xlim(0, 0.6); b.set_ylim(0, 4.2)
    b.set_xlabel('D'); b.set_ylabel('T')
    b.set_title('(b) diagrama (D,T): ferro (izq) / soft (der)'); b.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    N = 200
    N_SIM = 1000           # subir en produccion (>= 1000 para ~3% de error)

    # --- (a) <tau> vs D ---
    T_A = 0.75
    DA_RANGE = (0.05, 0.60, 10)
    Dv, dC = sweep_D_configs(linrange(*DA_RANGE), N=N, T_frac=T_A,
                             tau_max=20000, n_sim=N_SIM)

    # --- (b) malla y fronteras ---
    D_grid = linrange(0.05, 0.60, 9)
    T_grid = linrange(2.0, 4.0, 9)
    G = grid_tau(D_grid, T_grid, N=N, cap=1000000, n_sim=10)
    TAU_FAMILY = [10000, 100000, 1000000]

    make_figure(Dv, dC, T_A, D_grid, T_grid, G, TAU_FAMILY,
                'outputs/c3_stochastic.png')
    print('Figura: c3_stochastic.png')


if __name__ == '__main__':
    main()
