"""
C4 - Inversion con tres bloques (k = 3).

Repite los resultados clave de C1 para k = 3 y muestra el fenomeno nuevo de
k > 2: la inversion multietapa. Como m_total = (1/k) sum_b m_b, para cruzar
m_total = 0 desde el estado ordenado hace falta voltear una MAYORIA de bloques
(2 de 3), de modo que la inversion procede por etapas (un bloque voltea, meseta
cerca de -m*/3, vuelta a esperar, voltea el segundo y se cruza).

Caso base k=3 (un solo J_intra y un solo J_inter), con Tc = J_medio = 1:
    J_intra + 2 J_inter = 3. Se usa ACOPLAMIENTO DEBIL: J_intra = 2.7, J_inter = 0.15.
    (Distinto del resto de campanas: el acoplamiento debil hace mas visible el
    volteo secuencial de los bloques en la trayectoria de inversion.)
Comparacion a igual Tc = 1: Curie-Weiss (J=1) y bloque k=2 (J11=1.2, J12=0.8).

Genera dos figuras:
  c4_scaling.png    - <tau> vs T y vs N para CW, k=2 y k=3.
  c4_trajectory.png - trayectorias m_total(t) y m_b(t) en inversiones k=3.
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


def cw(N, T):
    return BlockIsingParams(N=N, n_blocks=1, proportions=[1.0], J_intra=1.0, T=T)


def k2(N, T):
    return BlockIsingParams(N=N, n_blocks=2, proportions=[0.5, 0.5],
                            J_intra=1.2, J_inter=0.8, T=T)


def k3(N, T):
    return BlockIsingParams(N=N, n_blocks=3, proportions=[1/3, 1/3, 1/3],
                            J_intra=2.7, J_inter=0.15, T=T)


CONFIGS = {
    'Curie-Weiss': (cw, 's--'),
    '$k=2$':       (k2, 'o-'),
    '$k=3$':       (k3, '^-'),
}


def mean_tau(params, tau_max, n_sim, seed=1):
    model = BlockIsingModel(params, seed=seed)
    r = reversal_time_ensemble(model, constant_field(0.0), tau_max, n_sim)
    return r['mean'], r['se']


# --------------------------------------------------------------------------
#  (1) Escalado <tau> vs T y vs N
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


def figure_scaling(Tf, dT, Nv, dN, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, (fac, sty) in CONFIGS.items():
        ax[0].errorbar(1 / Tf, dT[lab][0], yerr=dT[lab][1], fmt=sty, ms=4, capsize=2, label=lab)
        ax[1].errorbar(Nv, dN[lab][0], yerr=dN[lab][1], fmt=sty, ms=4, capsize=2, label=lab)
    ax[0].set_yscale('log'); ax[0].set_xlabel(r'$T_c/T$'); ax[0].set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    ax[0].set_title('(a) $\\langle\\tau\\rangle$ vs T'); ax[0].legend(fontsize=8)
    ax[1].set_yscale('log'); ax[1].set_xlabel('N'); ax[1].set_ylabel(r'$\langle\tau\rangle$ (MCS)')
    ax[1].set_title('(b) $\\langle\\tau\\rangle$ vs N'); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)


# --------------------------------------------------------------------------
#  (2) Trayectorias multietapa (k = 3)
# --------------------------------------------------------------------------

def figure_trajectory(params, max_mcs, path, seed=1, record_every=2):
    """Una trayectoria de inversion desde el estado ordenado m = -1
    (mismas condiciones iniciales que para medir tau). Generaliza a cualquier k."""
    model = BlockIsingModel(params, seed=seed)
    model.reset(m_init=-1.0)
    res = model.run(max_mcs, constant_field(0.0), record_every=record_every)
    t = res['time']

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, mb in enumerate(res['m_blocks']):
        ax.plot(t, mb, lw=0.8, alpha=0.7, label=f'$m_{{{k + 1}}}$')
    ax.plot(t, res['m_total'], 'k-', lw=2, label=r'$m_{\rm total}$')
    ax.axhline(0, color='r', ls='--', lw=0.8)
    ax.set_xlabel('t (MCS)'); ax.set_ylabel('magnetizacion')
    ax.set_title(f'C4 - Trayectoria de inversion (k={params.n_blocks})')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)


def main():
    N_SIM = 70
    TAU_MAX = 100000

    # (1) escalado
    Tf, dT = sweep_T(linrange(0.84, 0.98, 8), N=200, tau_max=TAU_MAX, n_sim=N_SIM)
    Nv, dN = sweep_N(linrange(50, 350, 8), T_frac=0.88, tau_max=TAU_MAX, n_sim=N_SIM)
    figure_scaling(Tf, dT, Nv, dN, 'outputs/c4_scaling.png')
    print('Figura: c4_scaling.png')

    # (2) una trayectoria de ejemplo (k=3, desde m=-1)
    figure_trajectory(k3(N=300, T=0.86), max_mcs=8000,
                      path='outputs/c4_trajectory.png')
    print('Figura: c4_trajectory.png')


if __name__ == '__main__':
    main()
