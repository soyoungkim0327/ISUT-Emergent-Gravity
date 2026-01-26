# -*- coding: utf-8 -*-
"""isuit_100_qumond_pm.py

A minimal, self-contained QUMOND (Quasi-linear MOND) particle--mesh solver
implemented using NumPy FFTs.

Why this exists
---------------
If you want the paper to claim a *conservative* completion (force derives from a scalar
potential), then the engine must compute accelerations as

    a(x) = -∇Φ(x)

QUMOND provides a practical route using *two* Poisson solves:

  1) Newtonian potential:
        ∇² Φ_N = 4π G ρ

  2) QUMOND potential:
        ∇² Φ = ∇·[ ν(|∇Φ_N|/a0) ∇Φ_N ]

With the common sign convention g_N ≡ -∇Φ_N (inward Newtonian field), this can be written as

        ∇² Φ = -∇·[ ν(|g_N|/a0) g_N ]

and the physical acceleration is a ≡ -∇Φ.

Implementation choices
----------------------
- Uses FFT Poisson solves with periodic boundary conditions.
- For an *approximately* isolated solution, you can choose a large box or use padding.
- Uses spectral derivatives (FFT-based gradients/divergence) for consistency.

This module is intentionally dependency-light (NumPy only).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

Array = np.ndarray


# =============================================================================
# Interpolation functions ν(y)
# =============================================================================

def nu_simple(y: Array, y_floor: float = 1e-30) -> Array:
    """"Simple" MOND ν(y) commonly used in rotation-curve work.

    ν(y) = 0.5 * (1 + sqrt(1 + 4/y))

    Asymptotics:
      y >> 1  -> ν -> 1
      y << 1  -> ν ~ 1/sqrt(y)
    """
    y_safe = np.maximum(y, y_floor)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y_safe))


def nu_standard(y: Array, y_floor: float = 1e-30) -> Array:
    """ν(y) corresponding to μ(x)=x/sqrt(1+x^2) under spherical inversion.

    ν(y) = sqrt( 0.5 * ( 1 + sqrt(1 + 4/y^2 ) ) )

    Asymptotics:
      y >> 1  -> ν -> 1
      y << 1  -> ν ~ 1/sqrt(y)
    """
    y_safe = np.maximum(y, y_floor)
    return np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y_safe**2))))


# =============================================================================
# Optional: μβ-family and ν(y) table factory (for solar-system/EFE sensitivity)
# =============================================================================

def mu_beta(x: Array, beta: float) -> Array:
    """Smooth μβ-family:

    μβ(x) = x / (1 + x^β)^{1/β}

    Notes
    -----
    * beta=1 reproduces the common 'simple' μ(x)=x/(1+x).
    * beta=2 reproduces the 'standard' μ(x)=x/sqrt(1+x^2).
    """
    x_safe = np.maximum(x, 0.0)
    beta_f = float(beta)
    if beta_f <= 0.0:
        raise ValueError("beta must be positive")
    return x_safe / (1.0 + x_safe**beta_f) ** (1.0 / beta_f)


def make_nu_from_mu_beta(
    beta: float,
    y_min: float = 1e-8,
    y_max: float = 1e8,
    n_grid: int = 4096,
    n_bisect: int = 80,
) -> Callable[[Array], Array]:
    """Return a ν(y) callable induced by μβ via the spherical inversion y = x μ(x).

    This is intended for validation/sensitivity scans (e.g., solar-system EFE quadrupole checks).
    For performance in 3D field solves, we precompute x(y) on a log-spaced grid and use 1D interpolation.
    """
    beta_f = float(beta)
    if beta_f <= 0.0:
        raise ValueError("beta must be positive")
    y_min_f = float(y_min)
    y_max_f = float(y_max)
    if not (0.0 < y_min_f < y_max_f):
        raise ValueError("require 0 < y_min < y_max")
    if int(n_grid) < 32:
        raise ValueError("n_grid too small")

    # Log-spaced y grid
    logy_grid = np.linspace(np.log10(y_min_f), np.log10(y_max_f), int(n_grid))
    y_grid = 10.0 ** logy_grid

    # Solve y = x μβ(x) for x>0 (vectorized bisection)
    x_low = np.zeros_like(y_grid)
    x_high = np.maximum(4.0 * y_grid, 4.0 * np.sqrt(y_grid))

    def f(x: Array) -> Array:
        return x * mu_beta(x, beta_f) - y_grid

    # Ensure upper bracket is above the root for all y
    fh = f(x_high)
    it_expand = 0
    while np.any(fh < 0.0) and it_expand < 20:
        x_high = np.where(fh < 0.0, x_high * 2.0, x_high)
        fh = f(x_high)
        it_expand += 1
    if np.any(fh < 0.0):
        raise RuntimeError("failed to bracket μβ inversion for some y; increase y_max or expansion")

    for _ in range(int(n_bisect)):
        x_mid = 0.5 * (x_low + x_high)
        fm = x_mid * mu_beta(x_mid, beta_f) - y_grid
        x_high = np.where(fm > 0.0, x_mid, x_high)
        x_low = np.where(fm > 0.0, x_low, x_mid)

    x_grid = 0.5 * (x_low + x_high)
    nu_grid = x_grid / y_grid

    def nu_of_y(y: Array) -> Array:
        y_arr = np.asarray(y, dtype=np.float64)
        y_safe = np.maximum(y_arr, 1e-30)
        logy = np.log10(y_safe)

        # Interpolate in log-space (flatten + reshape for speed)
        nu_interp = np.interp(
            logy.ravel(), logy_grid, nu_grid, left=np.nan, right=np.nan
        ).reshape(y_safe.shape)

        # Deep-MOND: ν ~ 1/sqrt(y)
        nu_deep = 1.0 / np.sqrt(y_safe)
        # High-y: ν ≈ 1 + (1/β) y^{-β}
        nu_high = 1.0 + (1.0 / beta_f) * (y_safe ** (-beta_f))

        nu_out = np.where(y_safe < y_min_f, nu_deep, nu_interp)
        nu_out = np.where(y_safe > y_max_f, nu_high, nu_out)
        nu_out = np.where(np.isfinite(nu_out), nu_out, nu_deep)
        return nu_out.astype(np.float64, copy=False)

    return nu_of_y

# =============================================================================
# Grid + FFT helpers
# =============================================================================

@dataclass
class PMGrid3D:
    """A simple cubic particle-mesh grid for FFT Poisson solves."""

    N: int
    boxsize: float
    G: float = 1.0

    # Note: we keep k-grids in float64 for numerical stability even if fields
    # are stored as float32.
    _kx: Array | None = None
    _ky: Array | None = None
    _kz: Array | None = None
    _k2: Array | None = None

    def __post_init__(self) -> None:
        if self.N <= 8:
            raise ValueError("N is too small; use at least 16.")
        if self.boxsize <= 0:
            raise ValueError("boxsize must be positive")

        dx = self.dx
        k = (2.0 * np.pi) * np.fft.fftfreq(self.N, d=dx)
        kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
        k2 = kx**2 + ky**2 + kz**2

        self._kx = kx.astype(np.float64, copy=False)
        self._ky = ky.astype(np.float64, copy=False)
        self._kz = kz.astype(np.float64, copy=False)
        self._k2 = k2.astype(np.float64, copy=False)

    @property
    def dx(self) -> float:
        return float(self.boxsize) / float(self.N)

    @property
    def kx(self) -> Array:
        assert self._kx is not None
        return self._kx

    @property
    def ky(self) -> Array:
        assert self._ky is not None
        return self._ky

    @property
    def kz(self) -> Array:
        assert self._kz is not None
        return self._kz

    @property
    def k2(self) -> Array:
        assert self._k2 is not None
        return self._k2

    def centered_coords(self) -> Tuple[Array, Array, Array]:
        """Return mesh coordinates (x,y,z) centered at 0 in [-L/2, L/2)."""
        dx = self.dx
        ax = (np.arange(self.N) - self.N / 2.0) * dx
        x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
        return x, y, z


def _poisson_solve_periodic_from_rhs(rhs: Array, grid: PMGrid3D) -> Tuple[Array, Array]:
    """Solve ∇²φ = rhs on a periodic grid using FFT.

    Returns (phi_real, phi_k) where phi_k is the complex Fourier field.
    The mean (k=0) mode is set to 0.
    """
    rhs_k = np.fft.fftn(rhs)
    k2 = grid.k2

    phi_k = np.zeros_like(rhs_k, dtype=np.complex128)
    mask = k2 > 0
    phi_k[mask] = -rhs_k[mask] / k2[mask]
    phi_k[0, 0, 0] = 0.0 + 0.0j

    phi = np.fft.ifftn(phi_k).real
    return phi, phi_k


def _grad_from_phi_k(phi_k: Array, grid: PMGrid3D) -> Tuple[Array, Array, Array]:
    """Spectral gradient: ∇φ from φ_k."""
    gx = np.fft.ifftn(1j * grid.kx * phi_k).real
    gy = np.fft.ifftn(1j * grid.ky * phi_k).real
    gz = np.fft.ifftn(1j * grid.kz * phi_k).real
    return gx, gy, gz


def _divergence_periodic(Fx: Array, Fy: Array, Fz: Array, grid: PMGrid3D) -> Array:
    """Spectral divergence: ∇·F for a periodic grid."""
    Fx_k = np.fft.fftn(Fx)
    Fy_k = np.fft.fftn(Fy)
    Fz_k = np.fft.fftn(Fz)

    div_k = 1j * (grid.kx * Fx_k + grid.ky * Fy_k + grid.kz * Fz_k)
    div = np.fft.ifftn(div_k).real
    return div


# =============================================================================
# CIC deposit/gather
# =============================================================================

def _wrap_to_unit_box(pos: Array, boxsize: float) -> Array:
    """Map positions from any real coordinates to [0, L) assuming periodicity.

    Input pos is (...,3)."""
    L = float(boxsize)
    return np.mod(pos + 0.5 * L, L)


def cic_deposit_density(
    pos: Array,
    mass: Array,
    grid: PMGrid3D,
    out: Optional[Array] = None,
    dtype: np.dtype = np.float64,
) -> Array:
    """Deposit particle masses to a density grid using CIC.

    Parameters
    ----------
    pos : (Np,3) particle positions in *centered* coordinates (recommended range [-L/2, L/2)).
    mass: (Np,) particle masses
    grid: PMGrid3D

    Returns
    -------
    rho : (N,N,N) density (mass per volume)
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be shape (Np,3)")
    if mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
        raise ValueError("mass must be shape (Np,)")

    N = grid.N
    dx = grid.dx
    L = grid.boxsize

    if out is None:
        rho = np.zeros((N, N, N), dtype=dtype)
    else:
        rho = out
        rho.fill(0.0)

    # Map to [0,L)
    u = _wrap_to_unit_box(pos, L) / dx  # in grid units

    i0 = np.floor(u).astype(np.int64)
    f = u - i0

    i0x, i0y, i0z = i0[:, 0], i0[:, 1], i0[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    # neighbor indices (periodic)
    i1x = (i0x + 1) % N
    i1y = (i0y + 1) % N
    i1z = (i0z + 1) % N

    w0x, w1x = (1.0 - fx), fx
    w0y, w1y = (1.0 - fy), fy
    w0z, w1z = (1.0 - fz), fz

    # convert mass to density contribution
    inv_cell_vol = 1.0 / (dx**3)
    m = mass.astype(np.float64, copy=False) * inv_cell_vol

    # 8 corners
    def add(ix, iy, iz, w):
        np.add.at(rho, (ix, iy, iz), m * w)

    add(i0x, i0y, i0z, w0x * w0y * w0z)
    add(i0x, i0y, i1z, w0x * w0y * w1z)
    add(i0x, i1y, i0z, w0x * w1y * w0z)
    add(i0x, i1y, i1z, w0x * w1y * w1z)

    add(i1x, i0y, i0z, w1x * w0y * w0z)
    add(i1x, i0y, i1z, w1x * w0y * w1z)
    add(i1x, i1y, i0z, w1x * w1y * w0z)
    add(i1x, i1y, i1z, w1x * w1y * w1z)

    return rho


def cic_gather_vector_field(
    field: Array,
    pos: Array,
    grid: PMGrid3D,
) -> Array:
    """Gather a vector field defined on the grid to particle positions via CIC.

    field: (N,N,N,3)
    pos: (Np,3) in centered coords
    returns: (Np,3)
    """
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("field must have shape (N,N,N,3)")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be shape (Np,3)")

    N = grid.N
    dx = grid.dx
    L = grid.boxsize

    u = _wrap_to_unit_box(pos, L) / dx
    i0 = np.floor(u).astype(np.int64)
    f = u - i0

    i0x, i0y, i0z = i0[:, 0], i0[:, 1], i0[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    i1x = (i0x + 1) % N
    i1y = (i0y + 1) % N
    i1z = (i0z + 1) % N

    w0x, w1x = (1.0 - fx), fx
    w0y, w1y = (1.0 - fy), fy
    w0z, w1z = (1.0 - fz), fz

    out = np.zeros((pos.shape[0], 3), dtype=np.float64)

    def acc(ix, iy, iz, w):
        out[:] += field[ix, iy, iz, :] * w[:, None]

    acc(i0x, i0y, i0z, w0x * w0y * w0z)
    acc(i0x, i0y, i1z, w0x * w0y * w1z)
    acc(i0x, i1y, i0z, w0x * w1y * w0z)
    acc(i0x, i1y, i1z, w0x * w1y * w1z)

    acc(i1x, i0y, i0z, w1x * w0y * w0z)
    acc(i1x, i0y, i1z, w1x * w0y * w1z)
    acc(i1x, i1y, i0z, w1x * w1y * w0z)
    acc(i1x, i1y, i1z, w1x * w1y * w1z)

    return out


def cic_gather_scalar_field(
    field: Array,
    pos: Array,
    grid: PMGrid3D,
) -> Array:
    """Gather a scalar field defined on the grid to particle positions via CIC.

    field: (N,N,N)
    pos: (Np,3) in centered coords
    returns: (Np,)
    """
    if field.ndim != 3:
        raise ValueError("field must have shape (N,N,N)")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be shape (Np,3)")

    N = grid.N
    dx = grid.dx
    L = grid.boxsize

    u = _wrap_to_unit_box(pos, L) / dx
    i0 = np.floor(u).astype(np.int64)
    f = u - i0

    i0x, i0y, i0z = i0[:, 0], i0[:, 1], i0[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    i1x = (i0x + 1) % N
    i1y = (i0y + 1) % N
    i1z = (i0z + 1) % N

    w0x, w1x = (1.0 - fx), fx
    w0y, w1y = (1.0 - fy), fy
    w0z, w1z = (1.0 - fz), fz

    out = np.zeros((pos.shape[0],), dtype=np.float64)

    def acc(ix, iy, iz, w):
        out[:] += field[ix, iy, iz] * w

    acc(i0x, i0y, i0z, w0x * w0y * w0z)
    acc(i0x, i0y, i1z, w0x * w0y * w1z)
    acc(i0x, i1y, i0z, w0x * w1y * w0z)
    acc(i0x, i1y, i1z, w0x * w1y * w1z)

    acc(i1x, i0y, i0z, w1x * w0y * w0z)
    acc(i1x, i0y, i1z, w1x * w0y * w1z)
    acc(i1x, i1y, i0z, w1x * w1y * w0z)
    acc(i1x, i1y, i1z, w1x * w1y * w1z)

    return out


# =============================================================================
# Analytic baryonic density helper (exponential disk)
# =============================================================================

def exponential_disk_density(
    grid: PMGrid3D,
    M_total: float,
    R_d: float,
    z_d: float,
    renormalize: bool = True,
) -> Array:
    """Create a 3D exponential disk density field.

    ρ(R,z) = (M_total/(4π R_d^2 z_d)) exp(-R/R_d) exp(-|z|/z_d)

    This is a simple analytic mass model used for demos/tests.

    Notes:
    - On a finite periodic box, some mass is truncated; if renormalize=True,
      we rescale the field so that the discrete sum equals M_total.
    """
    x, y, z = grid.centered_coords()
    R = np.sqrt(x**2 + y**2)

    rho0 = M_total / (4.0 * np.pi * (R_d**2) * z_d)
    rho = rho0 * np.exp(-R / max(R_d, 1e-12)) * np.exp(-np.abs(z) / max(z_d, 1e-12))

    if renormalize:
        m_discrete = float(np.sum(rho) * (grid.dx**3))
        if m_discrete > 0:
            rho *= (M_total / m_discrete)

    return rho


# =============================================================================
# QUMOND solver
# =============================================================================

@dataclass
class QUMONDResult:
    phi_N: Array
    phi: Array
    accel: Array  # (N,N,N,3)


class QUMONDSolverFFT:
    """FFT-based QUMOND solver on a periodic PM grid."""

    def __init__(self, grid: PMGrid3D):
        self.grid = grid

    def solve(
        self,
        rho: Array,
        a0: float,
        nu_func: Callable[[Array], Array] = nu_standard,
        pad_factor: int = 1,
        eps_g: float = 1e-30,
        g_ext: Optional[Tuple[float, float, float]] = None,
    ) -> QUMONDResult:
        """Solve QUMOND for a given baryonic density rho.

        Parameters
        ----------
        rho : (N,N,N)
            Mass density on the grid.
        a0 : float
            Acceleration scale.
        nu_func : callable
            ν(y) function.
        pad_factor : int
            If >1, zero-pad rho to (pad_factor*N)^3, solve, then crop.
            This approximates isolated boundaries by pushing periodic images farther away.
        eps_g : float
            Floor for |g_N| to avoid division-by-zero.
        g_ext : (3,) tuple or None
            Optional constant external acceleration vector to emulate the MOND External Field Effect
            by evaluating ν(|g_N + g_ext|/a0) in the QUMOND source term.
        """
        if rho.ndim != 3 or rho.shape[0] != rho.shape[1] or rho.shape[0] != rho.shape[2]:
            raise ValueError("rho must be a cubic 3D array")

        if pad_factor not in (1, 2, 3):
            raise ValueError("pad_factor supported values: 1,2,3")

        if pad_factor == 1:
            return self._solve_core(rho=rho, a0=a0, nu_func=nu_func, eps_g=eps_g, g_ext=g_ext)

        # padded solve
        N = rho.shape[0]
        Np = pad_factor * N
        Lp = pad_factor * self.grid.boxsize
        grid_p = PMGrid3D(N= Np, boxsize=Lp, G=self.grid.G)

        rho_p = np.zeros((Np, Np, Np), dtype=np.float64)
        start = (Np - N) // 2
        rho_p[start:start+N, start:start+N, start:start+N] = rho.astype(np.float64, copy=False)

        solver_p = QUMONDSolverFFT(grid_p)
        res_p = solver_p._solve_core(rho=rho_p, a0=a0, nu_func=nu_func, eps_g=eps_g, g_ext=g_ext)

        # crop back
        phiN = res_p.phi_N[start:start+N, start:start+N, start:start+N]
        phi = res_p.phi[start:start+N, start:start+N, start:start+N]
        accel = res_p.accel[start:start+N, start:start+N, start:start+N, :]
        return QUMONDResult(phi_N=phiN, phi=phi, accel=accel)

    def _solve_core(
        self,
        rho: Array,
        a0: float,
        nu_func: Callable[[Array], Array],
        eps_g: float,
        g_ext: Optional[Tuple[float, float, float]],
    ) -> QUMONDResult:
        grid = self.grid

        # 1) Newtonian Poisson: ∇² Φ_N = 4πG ρ
        rhs_N = 4.0 * np.pi * grid.G * rho
        phiN, phiN_k = _poisson_solve_periodic_from_rhs(rhs_N, grid)

        # g_N = -∇Φ_N
        dphiN_dx, dphiN_dy, dphiN_dz = _grad_from_phi_k(phiN_k, grid)
        gNx = -dphiN_dx
        gNy = -dphiN_dy
        gNz = -dphiN_dz

        # Optional External Field Effect (EFE): include a constant external field in ν(|g_N + g_ext|)
        if g_ext is not None:
            ge_x, ge_y, ge_z = g_ext
            gTx = gNx + float(ge_x)
            gTy = gNy + float(ge_y)
            gTz = gNz + float(ge_z)
        else:
            gTx, gTy, gTz = gNx, gNy, gNz

        gT_mag = np.sqrt(gTx**2 + gTy**2 + gTz**2)
        y = gT_mag / max(float(a0), eps_g)
        nu = nu_func(y)

        # 2) QUMOND source: ∇² Φ = -∇·[ ν(|g_N + g_ext|/a0) (g_N + g_ext) ] (reduces to standard QUMOND if g_ext=None)
        Fx = nu * gTx
        Fy = nu * gTy
        Fz = nu * gTz

        divF = _divergence_periodic(Fx, Fy, Fz, grid)
        rhs = -divF

        phi, phi_k = _poisson_solve_periodic_from_rhs(rhs, grid)

        # 3) a = -∇Φ
        dphi_dx, dphi_dy, dphi_dz = _grad_from_phi_k(phi_k, grid)
        ax = -dphi_dx
        ay = -dphi_dy
        az = -dphi_dz

        accel = np.stack([ax, ay, az], axis=-1)
        return QUMONDResult(phi_N=phiN, phi=phi, accel=accel)


# =============================================================================
# Convenience wrapper: static field sampling
# =============================================================================

@dataclass
class StaticFieldSampler:
    """Holds a static potential/acceleration field and provides CIC sampling."""

    grid: PMGrid3D
    phi: Array
    accel: Array  # (N,N,N,3)

    def accel_at(self, pos: Array) -> Array:
        return cic_gather_vector_field(self.accel, pos, self.grid)

    def phi_at(self, pos: Array) -> Array:
        return cic_gather_scalar_field(self.phi, pos, self.grid)