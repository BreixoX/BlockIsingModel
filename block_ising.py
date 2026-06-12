"""
Block Ising Model de campo medio (Curie-Weiss multicomponente) con K bloques.
Motor de simulacion: dinamica de Glauber + observables. Sin codigo de figuras.

Hamiltoniano (sin autointeraccion, i != j):
    H = -1/(2N) sum_{k,l} J_{kl} M_k M_l  -  h(t) sum_i sigma_i,
con M_k = sum_{i in B_k} sigma_i la magnetizacion total del bloque k.

Campo efectivo sobre un spin i del bloque k (con correccion de tamano finito):
    h_k^eff(i) = sum_l J_{kl} p_l m_l  +  h(t)  -  J_{kk} sigma_i / N,
con p_l = N_l/N. El ultimo termino, O(1/N), elimina la autointeraccion del
spin consigo mismo y solo importa en los barridos sobre N.

Dinamica de Glauber (bano termico):
    P(sigma_i = +1) = 1 / (1 + exp(-2 beta h_k^eff)).

Unidad de tiempo = 1 MCS = N actualizaciones elementales aleatorias.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Dict
import numpy as np
import numba


# --------------------------------------------------------------------------
#  Parametros del modelo
# --------------------------------------------------------------------------

@dataclass
class BlockIsingParams:
    """Parametros del Block Ising Model de campo medio.

    La matriz de interaccion se construye con J[k,k]=J_intra y J[k,l]=J_inter.
    Para una matriz J generica, asignar directamente el atributo `.J`.
    """
    N: int = 1000
    n_blocks: int = 2
    proportions: Optional[List[float]] = None
    J_intra: float = 1.0
    J_inter: float = 0.5
    T: float = 1.2

    def __post_init__(self):
        if self.proportions is None:
            self.proportions = [1.0 / self.n_blocks] * self.n_blocks
        assert abs(sum(self.proportions) - 1.0) < 1e-10, \
            "Las proporciones deben sumar 1."
        self.beta = 1.0 / self.T
        self.J = np.full((self.n_blocks, self.n_blocks), self.J_inter)
        np.fill_diagonal(self.J, self.J_intra)
        self.block_sizes = [int(p * self.N) for p in self.proportions]
        self.block_sizes[-1] = self.N - sum(self.block_sizes[:-1])

    @property
    def Tc(self) -> float:
        """Temperatura critica de campo medio (h=0): Tc = rho(D_p J).

        Se calcula como el mayor autovalor de la matriz simetrica semejante
        S = D_p^{1/2} J D_p^{1/2}, que tiene el mismo espectro que D_p J pero
        es simetrica (asi eigvalsh es exacto tambien con proporciones no
        uniformes). Convencion ferromagnetica / J definida positiva.
        """
        sp = np.sqrt(np.asarray(self.proportions))
        S = sp[:, None] * self.J * sp[None, :]
        return float(np.max(np.linalg.eigvalsh(S)))

    def info(self) -> str:
        return (f"N={self.N}, K={self.n_blocks}, J_intra={self.J_intra}, "
                f"J_inter={self.J_inter}, T={self.T:.3f} "
                f"(Tc={self.Tc:.3f}, T/Tc={self.T/self.Tc:.3f})")


# --------------------------------------------------------------------------
#  Protocolos de campo (solo los dos que usamos)
# --------------------------------------------------------------------------

def constant_field(h0: float = 0.0) -> Callable[[int], float]:
    """Campo estatico h(t) = h0."""
    def field(t: int) -> float:
        return h0
    return field


def gaussian_field(D: float) -> Callable[[int], float]:
    """Campo estocastico gaussiano (Toral et al. 2025):
        h(t) ~ N(0, 2D), remuestreado una vez por MCS.
    D es la intensidad del campo (no confundir con D_p, las proporciones).
    """
    sigma = np.sqrt(2.0 * D)

    def field(t: int) -> float:
        return np.random.normal(0.0, sigma)
    return field


# --------------------------------------------------------------------------
#  Nucleo JIT: un paso MC de dinamica de Glauber spin a spin
# --------------------------------------------------------------------------

@numba.njit(cache=True)
def _glauber_kernel(spins_flat, block_labels, J, p, m, n_sizes, beta, h_ext, N):
    """Un MCS: N actualizaciones elementales aleatorias con reemplazo.

    Para cada microactualizacion: elige i al azar, identifica su bloque k,
    calcula h_k^eff con las magnetizaciones actuales (mas la correccion de
    autointeraccion), reasigna sigma_i segun Glauber y actualiza m[k] en O(1).
    El campo h_ext se mantiene constante durante todo el MCS.
    """
    K = J.shape[0]
    for _ in range(N):
        i = np.random.randint(0, N)
        k = block_labels[i]

        h_eff = h_ext
        for l in range(K):
            h_eff += J[k, l] * p[l] * m[l]
        h_eff -= J[k, k] * spins_flat[i] / N   # correccion de tamano finito

        prob_up = 1.0 / (1.0 + np.exp(-2.0 * beta * h_eff))
        sigma_old = spins_flat[i]
        sigma_new = np.int64(1) if np.random.random() < prob_up else np.int64(-1)
        spins_flat[i] = sigma_new
        m[k] += (sigma_new - sigma_old) / n_sizes[k]
    return spins_flat, m


# --------------------------------------------------------------------------
#  Modelo
# --------------------------------------------------------------------------

class BlockIsingModel:
    """Simulacion Monte Carlo (Glauber) del Block Ising Model de campo medio."""

    def __init__(self, params: BlockIsingParams, seed: Optional[int] = None):
        self.p = params
        if seed is not None:
            np.random.seed(seed)
        self.block_labels = np.concatenate([
            np.full(n, k, dtype=np.int64)
            for k, n in enumerate(params.block_sizes)
        ])
        self._J = params.J.astype(np.float64)
        self._p = np.asarray(params.proportions, dtype=np.float64)
        self._nk = np.asarray(params.block_sizes, dtype=np.float64)
        self._init_spins()

    def _init_spins(self, m_init: Optional[float] = None):
        if m_init is None:
            raw = np.random.choice(np.array([-1, 1], dtype=np.int64), size=self.p.N)
        else:
            n_up = max(0, min(self.p.N, int(round((1 + m_init) / 2 * self.p.N))))
            raw = np.full(self.p.N, -1, dtype=np.int64)
            raw[:n_up] = 1
            np.random.shuffle(raw)
        self.spins_flat = raw
        splits = np.cumsum([0] + self.p.block_sizes)
        self.spins = [self.spins_flat[splits[k]:splits[k + 1]]
                      for k in range(self.p.n_blocks)]
        self._update_m()

    def _update_m(self):
        self.m = np.array([float(np.mean(s)) for s in self.spins])
        self.m_total = float(np.dot(self.p.proportions, self.m))

    def _glauber_step(self, h_ext: float):
        self.spins_flat, self.m = _glauber_kernel(
            self.spins_flat, self.block_labels, self._J, self._p, self.m,
            self._nk, self.p.beta, float(h_ext), self.p.N)
        self.m_total = float(np.dot(self.p.proportions, self.m))

    def reset(self, m_init: Optional[float] = None):
        self._init_spins(m_init)

    def thermalize(self, n_therm: int, field_func: Callable[[int], float]):
        for t in range(n_therm):
            self._glauber_step(field_func(t))

    def run(self, n_steps: int, field_func: Callable[[int], float],
            record_every: int = 1) -> Dict:
        """Ejecuta n_steps MCS registrando la evolucion temporal."""
        n_rec = n_steps // record_every
        times = np.empty(n_rec, dtype=np.int64)
        m_rec = np.empty(n_rec)
        h_rec = np.empty(n_rec)
        mb_rec = [np.empty(n_rec) for _ in range(self.p.n_blocks)]
        idx = 0
        for t in range(n_steps):
            h_t = field_func(t)
            self._glauber_step(h_t)
            if t % record_every == 0 and idx < n_rec:
                times[idx] = t
                m_rec[idx] = self.m_total
                h_rec[idx] = h_t
                for k in range(self.p.n_blocks):
                    mb_rec[k][idx] = self.m[k]
                idx += 1
        return {'time': times[:idx], 'm_total': m_rec[:idx],
                'm_blocks': [mb[:idx] for mb in mb_rec], 'h': h_rec[:idx]}


# --------------------------------------------------------------------------
#  Puntos fijos de campo medio
# --------------------------------------------------------------------------

def mean_field_fixed_points(params: BlockIsingParams,
                            h: float = 0.0) -> List[np.ndarray]:
    """Puntos fijos de m_k = tanh(beta (sum_l J_{kl} p_l m_l + h)).

    Iteracion amortiguada desde una rejilla de condiciones iniciales; se
    eliminan duplicados. (No clasifica estabilidad.)
    """
    import itertools
    p = np.asarray(params.proportions)
    fixed_points: List[np.ndarray] = []
    grid = np.linspace(-0.98, 0.98, 7)
    for m0 in itertools.product(grid, repeat=params.n_blocks):
        m = np.array(m0, dtype=float)
        for _ in range(2000):
            m_new = np.tanh(params.beta * (params.J @ (p * m) + h))
            if np.max(np.abs(m_new - m)) < 1e-9:
                break
            m = 0.7 * m_new + 0.3 * m
        residual = np.max(np.abs(np.tanh(params.beta * (params.J @ (p * m) + h)) - m))
        if residual < 1e-5 and all(np.max(np.abs(fp - m)) > 1e-3 for fp in fixed_points):
            fixed_points.append(m.copy())
    return fixed_points


# --------------------------------------------------------------------------
#  Observables
# --------------------------------------------------------------------------

def reversal_time(model: BlockIsingModel, field_func: Callable[[int], float],
                  tau_max: int):
    """Tiempo de inversion (definicion adoptada, primer paso):
        tau = inf{ t : m(t) >= 0 | m(0) = -1 }.

    Parte del estado completamente ordenado m = -1 (sin termalizacion) y
    devuelve (tau, reversed):
      - reversed=True : la inversion ocurrio; tau es el numero de MCS.
      - reversed=False: censurado (no hubo inversion antes de tau_max); tau=tau_max.
    """
    model.reset(m_init=-1.0)
    for t in range(tau_max):
        model._glauber_step(field_func(t))
        if model.m_total >= 0.0:
            return t + 1, True
    return tau_max, False


def reversal_time_ensemble(model: BlockIsingModel,
                           field_func: Callable[[int], float],
                           tau_max: int, n_sim: int) -> Dict:
    """n_sim realizaciones independientes de reversal_time.

    Devuelve un dict con el array de tiempos, la mascara de inversiones
    efectivas, la fraccion censurada y, sobre las inversiones efectivas,
    la media y el error estandar (SE = s/sqrt(n_eff)).
    """
    taus = np.empty(n_sim)
    rev = np.zeros(n_sim, dtype=bool)
    for s in range(n_sim):
        taus[s], rev[s] = reversal_time(model, field_func, tau_max)
    t_eff = taus[rev]
    n_eff = t_eff.size
    mean = float(np.mean(t_eff)) if n_eff else np.nan
    se = float(np.std(t_eff, ddof=1) / np.sqrt(n_eff)) if n_eff > 1 else np.nan
    return {'tau': taus, 'reversed': rev,
            'censored_fraction': float(np.mean(~rev)),
            'mean': mean, 'se': se, 'n_eff': n_eff}


def magnetization_samples(model: BlockIsingModel,
                          field_func: Callable[[int], float],
                          n_therm: int, n_steps: int,
                          record_every: int = 1,
                          m_init: Optional[float] = None) -> Dict:
    """Muestras estacionarias de la magnetizacion para histogramas P(m) (C0).

    Termaliza n_therm MCS y registra m(t) cada record_every MCS durante
    n_steps. Devuelve el dict de `run` (time, m_total, m_blocks, h).
    """
    model.reset(m_init)
    model.thermalize(n_therm, field_func)
    return model.run(n_steps, field_func, record_every=record_every)


# --------------------------------------------------------------------------
#  Prueba minima
# --------------------------------------------------------------------------

if __name__ == '__main__':
    params = BlockIsingParams(N=800, n_blocks=2, J_intra=2.0, J_inter=0.6, T=1.0)
    print(params.info())
    print("Puntos fijos (h=0):",
          [tuple(np.round(fp, 3)) for fp in mean_field_fixed_points(params)])
    model = BlockIsingModel(params, seed=2025)
    tau, rev = reversal_time(model, gaussian_field(D=0.1), tau_max=20000)
    print(f"Tiempo de inversion (D=0.1): tau={tau} MCS, reversed={rev}")
