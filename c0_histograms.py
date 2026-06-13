"""
C0 - Distribuciones de la magnetizacion (validacion + resultado de bloque).

(1) c0_total.png  - Histograma de la magnetizacion TOTAL m = sum_b p_b m_b,
    estilo Fig. SM3 de Toral et al. (2025) en campo medio (Tc^MF = 4, acoples
    x4). Replica el resultado de campo medio (Collet 2014; Curie-Weiss) y
    muestra que el modelo de bloque se comporta como Curie-Weiss en m_total:
      filas D = 0, 0.1 ; columnas N = 10^3, 10^4 ; curvas T = 3.0,3.5,4.0,4.5.

(2) c0_vector.png - Distribucion conjunta del VECTOR (m1, m2), propia del
    modelo de bloque (Knopfel et al. 2020, Teorema 2): su correlacion refleja
    J12, algo que Curie-Weiss no puede mostrar. A Tc = J_medio = 1 (J11=2-J12):
      filas T = 1.1 Tc (desordenado), 0.9 Tc (ordenado) ; columnas J12 = 0.2,0.5,0.8.

Histogramas simetrizados por la simetria m -> -m (exacta a D=0 y campo de
media cero).
"""

import numpy as np
import matplotlib.pyplot as plt
from block_ising import (BlockIsingParams, BlockIsingModel,
                         constant_field, gaussian_field, magnetization_samples)


# ==========================================================================
#  (1) Magnetizacion total  (estilo SM3, Tc = 4)
# ==========================================================================


def pm_total(N, T, D, n_therm, n_steps, bins, seed=1):
    field = constant_field(0.0) if D == 0 else gaussian_field(D)
    model = BlockIsingModel(
        BlockIsingParams(N=N, n_blocks=2, proportions=[0.5, 0.5], J_intra=4.8, J_inter=3.2, T=T),
        seed=seed
    )
    res = magnetization_samples(model, field, n_therm, n_steps,
                                record_every=1, m_init=0.0)
    m = np.concatenate([res['m_total']])
    dens, edges = np.histogram(m, bins=bins, range=(-1, 1), density=True)
    return 0.5 * (edges[:-1] + edges[1:]), dens


def figure_total(N_list, T_list, D_list, n_therm, n_steps, bins, J_intra, J_inter, path):
    fig, ax = plt.subplots(len(D_list), len(N_list), figsize=(10, 6.5), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(T_list)))
    for r, D in enumerate(D_list):
        ymax = 0.0
        for c, N in enumerate(N_list):
            a = ax[r, c]
            for T, col in zip(T_list, colors):
                x, y = pm_total(N, T, D, n_therm, n_steps, bins)
                a.plot(x, y, color=col, lw=1.4, label=f'T={T:.1f}')
                ymax = max(ymax, y.max())
            a.set_title(f'D={D},  N={N}', fontsize=10)
            if r == len(D_list) - 1:
                a.set_xlabel('m')
            if c == 0:
                a.set_ylabel('P(m)')
        for c in range(len(N_list)):
            ax[r, c].set_ylim(0, 1.08 * ymax)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f'C0 - Magnetizacion total P(m)  (J_intra={J_intra}, J_inter={J_inter}, n_steps={n_steps}, Tc = 4)', y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ==========================================================================
#  (2) Vector de magnetizaciones  (Tc = 1, varia J12)
# ==========================================================================



def joint_samples(N, T, J12, n_therm, n_steps, seed=1):
    model = BlockIsingModel(
        BlockIsingParams(N=N, n_blocks=2, proportions=[0.5, 0.5], J_intra=2 - J12, J_inter=J12, T=T), # Tc = J_medio = 1
        seed=seed
    )
    res = magnetization_samples(model, constant_field(0.0), n_therm, n_steps,
                                record_every=1, m_init=0.0)
    m1, m2 = res['m_blocks']
    return m1, m2, np.corrcoef(m1, m2)[0, 1]


def figure_vector(N, T_rows, J12_cols, n_therm, n_steps, bins, path):
    fig, ax = plt.subplots(len(T_rows), len(J12_cols), figsize=(11, 7.2),
                           sharex=True, sharey=True)
    edges = np.linspace(-1, 1, bins + 1)
    for r, (Tf, reg) in enumerate(T_rows):
        for c, J12 in enumerate(J12_cols):
            a = ax[r, c]
            m1, m2, rho = joint_samples(N, Tf * 1.0, J12, n_therm, n_steps)
            H, _, _ = np.histogram2d(m1, m2, bins=[edges, edges], density=True)
            a.pcolormesh(edges, edges, H.T, cmap='viridis', shading='auto')
            a.plot([-1, 1], [-1, 1], 'w:', lw=0.8)
            a.set_title(f'$J_{{12}}={J12}$   ($\\rho={rho:.2f}$)', fontsize=9)
            a.set_aspect('equal')
            if r == len(T_rows) - 1:
                a.set_xlabel('$m_1$')
            if c == 0:
                a.set_ylabel(f'{reg}\n($T={Tf:.1f}\\,T_c$)\n\n$m_2$')
    fig.suptitle(f'C0 - Vector de magnetizaciones $P(m_1, m_2)$  (J_intra=2-J_{{12}}, n_steps={n_steps}, $T_c = 1$)', y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)

def main():
    # (1) magnetizacion total
    figure_total(N_list=[1000, 10000], T_list=[3.0, 3.5, 4.0, 4.5],
                 D_list=[0.0, 0.1], n_therm=2000, n_steps=100000, bins=61,
                 J_intra=4.8, J_inter=3.2,
                 path='outputs/c0_total.png')
    print('Figura: c0_total.png')


    # (2) vector de magnetizaciones
    figure_vector(N=200, T_rows=[(1.1, 'desordenado'), (0.9, 'ordenado')],
                  J12_cols=[0.2, 0.5, 0.8], n_therm=3000, n_steps=40000, bins=40,
                  path='outputs/c0_vector.png')
    print('Figura: c0_vector.png')


if __name__ == '__main__':
    main()
