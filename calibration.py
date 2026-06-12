"""
Calibracion del caso base: verificar la temperatura critica.

Comprueba que la magnetizacion observada en la simulacion Monte Carlo, |m|(T),
sigue la rama estable de campo medio y se anula en la Tc predicha por la formula
analitica Tc = rho(D_p J).

Caso base: J11=J22=1.2, J12=0.8, p=(1/2,1/2)  ->  J_medio=1.0, Tc=1.0.
"""

import numpy as np
import matplotlib.pyplot as plt
from block_ising import (BlockIsingParams, BlockIsingModel,
                         constant_field, mean_field_fixed_points)


def base_params(N: int, T: float) -> BlockIsingParams:
    return BlockIsingParams(N=N, n_blocks=2, proportions=[0.5, 0.5],
                            J_intra=1.2, J_inter=0.8, T=T)


def mean_field_order(T_values: np.ndarray) -> np.ndarray:
    """|m_total| de la rama ordenada de campo medio (h=0) para cada T.

    Toma el mayor |m_total| entre los puntos fijos de m_k = tanh(beta J(p o m)):
    por debajo de Tc da m*>0; por encima, solo existe m=0.
    """
    p = np.array([0.5, 0.5])
    order = np.zeros_like(T_values)
    for i, T in enumerate(T_values):
        fps = mean_field_fixed_points(base_params(1, T), h=0.0)
        order[i] = max(abs(float(np.dot(p, m))) for m in fps)
    return order


def mc_order(T_values: np.ndarray, N: int,
             n_therm: int, n_steps: int, every: int, seed: int = 0) -> np.ndarray:
    """<|m|>(T) en Monte Carlo, termalizando desde el estado ordenado."""
    mabs = np.zeros_like(T_values)
    for i, T in enumerate(T_values):
        model = BlockIsingModel(base_params(N, T), seed=seed)
        model.reset(m_init=0.9)
        model.thermalize(n_therm, constant_field(0.0))
        res = model.run(n_steps, constant_field(0.0), record_every=every)
        mabs[i] = float(np.mean(np.abs(res['m_total'])))
    return mabs


def main():
    Tc = base_params(1, 1.0).Tc
    print(f"Tc analitico (rho(D_p J)) = {Tc:.4f}")

    T_mf = np.linspace(0.3, 1.4, 45)
    order_mf = mean_field_order(T_mf)

    T_mc = np.linspace(0.3, 1.4, 23)
    mc = {N: mc_order(T_mc, N, n_therm=300, n_steps=2000, every=4, seed=7)
          for N in [1000, 4000, 10000]}
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(T_mf, order_mf, '-', color='k', lw=1.6, label='campo medio')
    for N, y in mc.items():
        ax.plot(T_mc, y, 'o', ms=4, alpha=0.85, label=f'MC  N={N}')
    ax.axvline(Tc, color='C3', ls='--', lw=1, label=f'$T_c={Tc:.2f}$')
    ax.set_xlabel('T'); ax.set_ylabel(r'$|m|$')
    ax.set_title('Verificación de $T_c$: magnetización vs temperatura')
    ax.legend()
    fig.tight_layout()
    fig.savefig('outputs/calibration.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print("Figura: calibration.png")


if __name__ == '__main__':
    main()
