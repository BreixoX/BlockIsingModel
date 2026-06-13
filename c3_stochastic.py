"""
C3 - Tiempo de inversion con campo estocastico gaussiano (Toral et al. 2025).

Campo espacialmente uniforme h(t) ~ N(0, 2D), remuestreado una vez por MCS.
Para que el rango D in [0, 0.6] sea informativo se usa el modelo reescalado a
Tc = 4 (acoplamientos x4), que coincide con la descripcion de campo medio de
Toral (numero de coordinacion 4, Tc^MF = 4; anexos SM3-SM5).

  (a) <tau> vs D a T fija (estilo Naskar: tau decrece con D), tres modelos de
      igual Tc = J_medio = 4: Curie-Weiss, bloque J12=3.2 (base), bloque J12=1.6.
  (b) Diagrama (D,T): frontera soft-ferro / ferro para una familia de tau_max,
      obtenida con el procedimiento de Toral (Fig. 4 / Fig. SM5): para cada D y
      cada tau_max se busca la PRIMERA temperatura a la que el sistema logra
      invertir (de m=-1 a m>=0) en menos de tau_max MCS. Esa temperatura es la
      frontera; por debajo el sistema es ferromagnetico (no invierte en tau_max,
      region ferro), por encima es soft-ferromagnetico (si invierte).

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
#  (a) <tau> vs D                                          (sin cambios)
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
#  (b) frontera soft-ferro / ferro segun el procedimiento de Toral
# --------------------------------------------------------------------------
#
#  Idea clave (eficiencia): una unica simulacion de longitud tau_big = max(
#  tau_family) devuelve el tiempo de inversion tau de cada realizacion. Para
#  CUALQUIER tau_max <= tau_big, "invierte en menos de tau_max" equivale a
#  "(invirtio) y (tau <= tau_max)". Asi se resuelve toda la familia de tau_max
#  con un solo barrido. Para cada (D, T) se estima la probabilidad de inversion
#  en tau_max con R realizaciones; la frontera T*(D) es la temperatura a la que
#  esa probabilidad cruza 1/2.
#
#  El barrido en T se hace DE ARRIBA (cerca de Tc) HACIA ABAJO y se detiene poco
#  despues de que ni el mayor tau_max consiga invertir: esto acota el numero de
#  simulaciones censuradas (las caras, de longitud completa) a unas pocas por D.

def reversal_fracs(N, T, D, tau_big, R, tau_family, seed):
    """Fraccion de R realizaciones que invierten en menos de cada tau_max."""
    np.random.seed(seed)                       # reproducibilidad del campo
    model = BlockIsingModel(block_params(N, T), seed=seed)
    res = reversal_time_ensemble(model, gaussian_field(D), tau_big, R)
    tau, rev = res['tau'], res['reversed']
    return [float(np.mean(rev & (tau <= tm))) for tm in tau_family]


def scan_T_down(D, N, T_top, T_bot, dT, tau_family, R, seed0,
                stop_eps=1e-9, stop_after=2):
    """Barre T de T_top a T_bot midiendo p(invierte en tau_max) en cada T.
    Se detiene stop_after pasos despues de que el mayor tau_max deje de
    invertir (ya por debajo de todas las fronteras)."""
    tau_family = sorted(tau_family)
    tau_big = max(tau_family)
    Ts, P = [], []
    n_zero, T, seed = 0, T_top, seed0
    while T >= T_bot - 1e-9:
        fr = reversal_fracs(N, T, D, tau_big, R, tau_family, seed)
        Ts.append(T); P.append(fr); seed += 1
        if fr[-1] <= stop_eps:                 # mayor tau_max ya no invierte
            n_zero += 1
            if n_zero >= stop_after:
                break
        else:
            n_zero = 0
        T -= dT
    return np.array(Ts), np.array(P)


def crossing(Ts_asc, p_asc, level=0.5):
    """T donde p(T) (creciente con T) cruza 'level'; None si no cruza."""
    for i in range(len(Ts_asc) - 1):
        a, b = p_asc[i], p_asc[i + 1]
        if a < level <= b:
            f = (level - a) / (b - a) if b != a else 0.0
            return Ts_asc[i] + f * (Ts_asc[i + 1] - Ts_asc[i])
    return None


def extract_boundaries(Ts, P, tau_family, level=0.5):
    order = np.argsort(Ts)                      # ascendente en T
    Ts, P = Ts[order], P[order]
    return {tm: crossing(Ts, P[:, k], level)
            for k, tm in enumerate(sorted(tau_family))}


def compute_boundaries(D_grid, N, T_top, T_bot, dT, tau_family, R, seed0=1000):
    """Para cada D y cada tau_max, T* de la frontera soft-ferro / ferro."""
    tau_family = sorted(tau_family)
    acc = {tm: ([], []) for tm in tau_family}
    seed = seed0
    for D in D_grid:
        Ts, P = scan_T_down(D, N, T_top, T_bot, dT, tau_family, R, seed)
        seed += 1000
        bnd = extract_boundaries(Ts, P, tau_family)
        for tm in tau_family:
            if bnd[tm] is not None:
                acc[tm][0].append(D); acc[tm][1].append(bnd[tm])
    return {tm: (np.array(a), np.array(b)) for tm, (a, b) in acc.items()}


# --------------------------------------------------------------------------
#  Figura
# --------------------------------------------------------------------------

def make_figure(Dv, dC, T_a, boundaries, path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))

    a = ax
    for lab, (fac, sty) in CONFIGS.items():
        y, e = dC[lab]
        a.errorbar(Dv, y, yerr=e, fmt=sty, ms=4, capsize=2, label=lab)
    a.set_yscale('log'); a.set_xlabel('D'); a.set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    a.set_title(f'(a) $\\langle\\tau\\rangle$ vs D   ($T={T_a:.2f}\\,T_c$)'); a.legend(fontsize=8)

    # Panel (b) comentado: frontera soft-ferro / ferro
    # b = ax[1]
    # for k, tm in enumerate(sorted(boundaries)):
    #     Ds, Ts = boundaries[tm]
    #     if len(Ds):
    #         b.plot(Ds, Ts, 'o-', ms=4, color=f'C{k}',
    #                label=r'$\tau_{max}=10^{%d}$' % int(round(np.log10(tm))))
    # b.axhline(TC, color='k', ls='--', lw=1, label=r'$T_c^{MF}=4$')
    # b.set_xlim(0, 0.6); b.set_ylim(0, 4.2)
    # b.set_xlabel('D'); b.set_ylabel('T')
    # b.set_title('(b) frontera: soft-ferro (arriba) / ferro (abajo)')
    # b.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    N = 1000
    N_SIM = 1000           # subir en produccion (>= 1000 para ~3% de error)

    # --- (a) <tau> vs D ---
    T_A = 0.75
    DA_RANGE = (0.05, 0.60, 10)
    Dv, dC = sweep_D_configs(linrange(*DA_RANGE), N=N, T_frac=T_A,
                             tau_max=20000, n_sim=N_SIM)

    # --- (b) frontera soft-ferro / ferro (procedimiento de Toral) 
    #TAU_FAMILY = [10, 100, 1000, 10000, 100000]
    #D_GRID = np.linspace(0.0, 0.6, 13)
    # boundaries = compute_boundaries(
    #     D_GRID, N=N, T_top=0.98 * TC, T_bot=0.1, dT=0.15,
    #     tau_family=TAU_FAMILY, R=10)

    make_figure(Dv, dC, T_A, {}, 'outputs/c3_stochastic.png')
    print('Figura: c3_stochastic.png')


if __name__ == '__main__':
    main()
