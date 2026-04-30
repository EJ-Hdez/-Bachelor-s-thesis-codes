# ════════════════════════════════════════════════════════════════
# LattEND.jl  v2.7.0
# Lattice END — Simulación 3D de colisiones atómicas/iónicas
#
# Electrón cuántico en malla 3D (CN-ADI + Strang splitting)
# Núcleos clásicos (Velocity-Verlet) con fuerzas:
#   (i)  Coulomb núcleo–núcleo
#   (ii) fuerza promedio electrón–núcleo (Ehrenfest)
# Máscara absorbente cos^(1/8) (Krause et al.)
#
# Esquema de propagación temporal (un paso completo Δt):
#   1. Nuclear VV (Δt/2)       — medio paso clásico
#   2. exp(-i V Δt/2)          — medio paso potencial electrónico
#   3. CN-ADI: alternante      — paso cinético implícito
#      Impares: z→y→x | Pares: x→y→z (cancela sesgo ADI a O(Δt²))
#   4. exp(-i V Δt/2)          — medio paso potencial electrónico
#   5. Nuclear VV (Δt/2)       — medio paso clásico
#   6. Máscara absorbente
#
# Historial:
#   v2.0–2.4 — Arquitectura struct, ADI alternante, energía, paro, salida
#   v2.5.0   — Diagnósticos normalizados (T_e_norm, E_total_norm)
#   v2.6.0   — Ajuste asintótico Ne(t) = β exp(-αt) + N∞ → Pi = 1-N∞
#   v2.7.0   — S_TRAJ, geometría flexible (axis,sign)+B_DIR, sin snap,
#              sin BOUNCE/EXIT/STALL, guarda de ajuste, Pion_raw col 19
# ════════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Dates
using DelimitedFiles

const VERSION_STR = "2.7.0"

# ════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DEL USUARIO
# ════════════════════════════════════════════════════════════════

const USER_NOTE = "Colisión p̄ + He⁺ a 90 eV "
const BASE_PATH = "/media/user/your/path/here/results"

# ── Sistema físico ──────────────────────────────────────────────
const ZT = 2.0                      # carga nuclear del target
const ZP = -1.0                     # carga nuclear del proyectil
const M_TARGET = 7294.299665        # masa del target (m_e)
const M_PROJ   = 1836.152673426     # masa del proyectil (m_e)
const M_EFF    = 1.0                # masa electrónica (u.a.)

# ── Dominio y malla ────────────────────────────────────────────
const XMIN, XMAX, NX = -17.1, 17.1, 170
const YMIN, YMAX, NY = -17.1, 17.1, 170
const ZMIN, ZMAX, NZ = -24.1, 24.1, 240

# ── Máscara absorbente ─────────────────────────────────────────
const LX_ABS = 5.5
const LY_ABS = 5.5
const LZ_ABS = 5.5

# ── Paso temporal ──────────────────────────────────────────────
const USE_COURANT    = true
const C_COURANT      = 0.50
const DT_MANUAL      = 0.08
const USE_VEL_LIMIT  = true
const C_VEL          = 0.30

# ── Geometría de colisión ──────────────────────────────────────
# DIRECTIONS: lista de (eje, signo). El signo define la dirección
#   del momento inicial del proyectil.
#   (:z, +1) → proyectil se mueve en +z
#   (:z, -1) → proyectil se mueve en -z
# B_DIR: eje perpendicular donde se coloca el parámetro de impacto.
const DIRECTIONS = [(:z, +1)]
const B_DIR      = :y

const B_LIST = [0.4]
#0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
#1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0
const E_KEV_LIST = [0.09]

# ── Trayectoria del proyectil ──────────────────────────────────
# S_TRAJ[1]: posición de salida del proyectil
# S_TRAJ[2]: posición de llegada (la simulación para al cruzar)
# T_classical = |S_TRAJ[2] - S_TRAJ[1]| / v₀ (automático)
const R_TARGET_INIT = [0.0, 0.0, 0.0]
const S_TRAJ    = [-9.0, 20.0]
const TMAX_HARD = 500.0    # tope absoluto de tiempo (u.a.)

# ── Plano de partición Ω ──────────────────────────────────────
const OMEGA_MODE  = :fixed
const OMEGA_FIXED = 0.0
const OMEGA_SHIFT = 0.0

# ── Regularización Coulomb ─────────────────────────────────────
const USE_GRID_RCUT = true
const RCUT_FACTOR   = 0.25

# ── Selección de proceso ───────────────────────────────────────
const PROCESS_MODE = :ionize

# ── Opciones de salida ─────────────────────────────────────────
const WRITE_TRAJ         = true
const WRITE_DENSITY      = true
const WRITE_ENERGY_LOG   = true
const ENERGY_LOG_EVERY   = 5     # cada N pasos; 0 = desactivar

# ════════════════════════════════════════════════════════════════
#  TIPOS / STRUCTS
# ════════════════════════════════════════════════════════════════

struct PhysSystem
    ZT::Float64; ZP::Float64
    M_target::Float64; M_proj::Float64
    M_eff::Float64
end

struct Grid3D
    X::Vector{Float64}; Y::Vector{Float64}; Z::Vector{Float64}
    Nx::Int; Ny::Int; Nz::Int
    Δx::Float64; Δy::Float64; Δz::Float64
    dV::Float64
end

struct KineticOps
    Hx_diag::Vector{Float64}; Hx_off::Vector{Float64}
    Hy_diag::Vector{Float64}; Hy_off::Vector{Float64}
    Hz_diag::Vector{Float64}; Hz_off::Vector{Float64}
end

# ════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE UNIDADES
# ════════════════════════════════════════════════════════════════

const HARTREE_TO_EV = 27.211386245988

function momentum_from_keV(E_keV::Float64, M_proj::Float64)
    E_Ha = (E_keV * 1000.0) / HARTREE_TO_EV
    return sqrt(2.0 * M_proj * E_Ha)
end

# ════════════════════════════════════════════════════════════════
#  MALLA
# ════════════════════════════════════════════════════════════════

function build_grid(xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz)
    Δx = (xmax - xmin) / (Nx + 1)
    Δy = (ymax - ymin) / (Ny + 1)
    Δz = (zmax - zmin) / (Nz + 1)
    X = [xmin + i * Δx for i in 0:Nx+1]
    Y = [ymin + j * Δy for j in 0:Ny+1]
    Z = [zmin + k * Δz for k in 0:Nz+1]
    return Grid3D(X, Y, Z, Nx, Ny, Nz, Δx, Δy, Δz, Δx * Δy * Δz)
end

# ════════════════════════════════════════════════════════════════
#  HAMILTONIANOS CINÉTICOS 1D
# ════════════════════════════════════════════════════════════════

function build_kinetic_1d(M_eff::Float64, N::Int, Δ::Float64)
    m = 0.5 / M_eff
    diag = fill(2m / Δ^2, N)
    off  = fill(-m / Δ^2, N - 1)
    return diag, off
end

function build_kinetic_ops(sys::PhysSystem, g::Grid3D)
    Hx_d, Hx_o = build_kinetic_1d(sys.M_eff, g.Nx, g.Δx)
    Hy_d, Hy_o = build_kinetic_1d(sys.M_eff, g.Ny, g.Δy)
    Hz_d, Hz_o = build_kinetic_1d(sys.M_eff, g.Nz, g.Δz)
    return KineticOps(Hx_d, Hx_o, Hy_d, Hy_o, Hz_d, Hz_o)
end

# ════════════════════════════════════════════════════════════════
#  MÁSCARA ABSORBENTE cos^(1/8)
# ════════════════════════════════════════════════════════════════

function build_1d_mask(coords::Vector{Float64}, N::Int,
                       u_min::Float64, u_max::Float64, L_abs::Float64)
    mask = ones(Float64, N)
    L_abs <= 0 && return mask
    L = min(L_abs, (u_max - u_min) / 2)
    @inbounds for i in 1:N
        u = coords[i+1]
        m = 1.0
        if abs(u_min - u) < L
            θ = π * abs(u - u_min + L) / (2L)
            m *= abs(cos(θ))^(1/8)
        end
        if abs(u_max - u) < L
            θ = π * abs(u - u_max + L) / (2L)
            m *= abs(cos(θ))^(1/8)
        end
        mask[i] = m
    end
    return mask
end

function build_absorbing_mask(g::Grid3D, Lx, Ly, Lz,
                              xmin, xmax, ymin, ymax, zmin, zmax)
    mx = build_1d_mask(g.X, g.Nx, xmin, xmax, Lx)
    my = build_1d_mask(g.Y, g.Ny, ymin, ymax, Ly)
    mz = build_1d_mask(g.Z, g.Nz, zmin, zmax, Lz)
    M = Array{Float64,3}(undef, g.Nx, g.Ny, g.Nz)
    @inbounds for ix in 1:g.Nx, iy in 1:g.Ny, iz in 1:g.Nz
        M[ix,iy,iz] = mx[ix] * my[iy] * mz[iz]
    end
    return M
end

function apply_mask!(ψ::Array{ComplexF64,3}, M::Array{Float64,3})
    @inbounds for I in CartesianIndices(ψ)
        ψ[I] *= M[I]
    end
end

# ════════════════════════════════════════════════════════════════
#  ESTADO INICIAL: 1s hidrogenoide
# ════════════════════════════════════════════════════════════════

function init_psi_1s!(ψ::Array{ComplexF64,3}, g::Grid3D,
                      R_T::Vector{Float64}, ZT::Float64)
    Nx, Ny, Nz = size(ψ)
    pref = ZT^(3/2) / sqrt(π)
    @inbounds for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
        dx = g.X[ix+1] - R_T[1]
        dy = g.Y[iy+1] - R_T[2]
        dz = g.Z[iz+1] - R_T[3]
        r = sqrt(dx*dx + dy*dy + dz*dz)
        ψ[ix,iy,iz] = pref * exp(-ZT * r) + 0im
    end
end

function normalize!(ψ::Array{ComplexF64,3}, dV::Float64)
    nrm = sum(abs2, ψ) * dV
    ψ ./= sqrt(nrm)
end

# ════════════════════════════════════════════════════════════════
#  POTENCIAL Y FUERZAS
# ════════════════════════════════════════════════════════════════

function potential!(V::Array{Float64,3}, g::Grid3D,
                    R::Matrix{Float64}, sys::PhysSystem, rcut2::Float64)
    Nx, Ny, Nz = size(V)
    ZT, ZP = sys.ZT, sys.ZP
    @inbounds for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
        x = g.X[ix+1]; y = g.Y[iy+1]; z = g.Z[iz+1]
        dxT = x - R[1,1]; dyT = y - R[1,2]; dzT = z - R[1,3]
        rT2 = max(dxT*dxT + dyT*dyT + dzT*dzT, rcut2)
        dxP = x - R[2,1]; dyP = y - R[2,2]; dzP = z - R[2,3]
        rP2 = max(dxP*dxP + dyP*dyP + dzP*dzP, rcut2)
        V[ix,iy,iz] = -ZT / sqrt(rT2) - ZP / sqrt(rP2)
    end
end

function nuc_forces!(F::Matrix{Float64}, R::Matrix{Float64}, sys::PhysSystem)
    dx = R[1,1]-R[2,1]; dy = R[1,2]-R[2,2]; dz = R[1,3]-R[2,3]
    r2 = max(dx*dx + dy*dy + dz*dz, 1e-24)
    r = sqrt(r2)
    coef = sys.ZT * sys.ZP / (r2 * r)
    F[1,1]= coef*dx;  F[1,2]= coef*dy;  F[1,3]= coef*dz
    F[2,1]=-coef*dx;  F[2,2]=-coef*dy;  F[2,3]=-coef*dz
end

function electron_forces!(F_e::Matrix{Float64}, ψ::Array{ComplexF64,3},
                          g::Grid3D, R::Matrix{Float64},
                          sys::PhysSystem, rcut2::Float64)
    Nx, Ny, Nz = size(ψ)
    ZT, ZP = sys.ZT, sys.ZP
    F_e .= 0.0
    @inbounds for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
        ρ = abs2(ψ[ix,iy,iz])
        ρ == 0.0 && continue
        x = g.X[ix+1]; y = g.Y[iy+1]; z = g.Z[iz+1]
        rxT = x-R[1,1]; ryT = y-R[1,2]; rzT = z-R[1,3]
        rT2 = max(rxT*rxT + ryT*ryT + rzT*rzT, rcut2)
        invT3 = 1.0 / (rT2 * sqrt(rT2))
        F_e[1,1] += ZT*ρ*rxT*invT3
        F_e[1,2] += ZT*ρ*ryT*invT3
        F_e[1,3] += ZT*ρ*rzT*invT3
        rxP = x-R[2,1]; ryP = y-R[2,2]; rzP = z-R[2,3]
        rP2 = max(rxP*rxP + ryP*ryP + rzP*rzP, rcut2)
        invP3 = 1.0 / (rP2 * sqrt(rP2))
        F_e[2,1] += ZP*ρ*rxP*invP3
        F_e[2,2] += ZP*ρ*ryP*invP3
        F_e[2,3] += ZP*ρ*rzP*invP3
    end
    @. F_e *= g.dV
end

# ════════════════════════════════════════════════════════════════
#  VELOCITY-VERLET
# ════════════════════════════════════════════════════════════════

function classical_step!(R, P, masses, τ, Fbuf, force_fn!)
    force_fn!(Fbuf, R)
    @inbounds for i in 1:2, j in 1:3; P[i,j] += 0.5*τ*Fbuf[i,j]; end
    @inbounds for i in 1:2, j in 1:3; R[i,j] += τ*P[i,j]/masses[i]; end
    force_fn!(Fbuf, R)
    @inbounds for i in 1:2, j in 1:3; P[i,j] += 0.5*τ*Fbuf[i,j]; end
end

# ════════════════════════════════════════════════════════════════
#  PROPAGADOR ELECTRÓNICO: CN-ADI
# ════════════════════════════════════════════════════════════════

function expV!(ψ::Array{ComplexF64,3}, V::Array{Float64,3}, half_dt::Float64)
    @inbounds for I in CartesianIndices(ψ)
        ψ[I] *= cis(-half_dt * V[I])
    end
end

function rhs_cn_1d!(dest, diag, off, ψl, cdt)
    N = length(diag)
    @inbounds begin
        dest[1] = diag[1]*ψl[1] + off[1]*ψl[2]
        for i in 2:N-1
            dest[i] = off[i-1]*ψl[i-1] + diag[i]*ψl[i] + off[i]*ψl[i+1]
        end
        dest[N] = off[N-1]*ψl[N-1] + diag[N]*ψl[N]
        @. dest = ψl - cdt*dest
    end
end

function build_cn_lu(diag, off, cdt)
    N = length(diag)
    lu(Tridiagonal(cdt .* off, ones(ComplexF64, N) .+ cdt .* diag, cdt .* off))
end

function sweep_x!(ψ, LU, diag, off, cdt)
    Nx, Ny, Nz = size(ψ)
    rhs = Vector{ComplexF64}(undef, Nx); line = similar(rhs)
    @inbounds for iz in 1:Nz, iy in 1:Ny
        for ix in 1:Nx; line[ix] = ψ[ix,iy,iz]; end
        rhs_cn_1d!(rhs, diag, off, line, cdt)
        line .= LU \ rhs
        for ix in 1:Nx; ψ[ix,iy,iz] = line[ix]; end
    end
end

function sweep_y!(ψ, LU, diag, off, cdt)
    Nx, Ny, Nz = size(ψ)
    rhs = Vector{ComplexF64}(undef, Ny); line = similar(rhs)
    @inbounds for iz in 1:Nz, ix in 1:Nx
        for iy in 1:Ny; line[iy] = ψ[ix,iy,iz]; end
        rhs_cn_1d!(rhs, diag, off, line, cdt)
        line .= LU \ rhs
        for iy in 1:Ny; ψ[ix,iy,iz] = line[iy]; end
    end
end

function sweep_z!(ψ, LU, diag, off, cdt)
    Nx, Ny, Nz = size(ψ)
    rhs = Vector{ComplexF64}(undef, Nz); line = similar(rhs)
    @inbounds for iy in 1:Ny, ix in 1:Nx
        for iz in 1:Nz; line[iz] = ψ[ix,iy,iz]; end
        rhs_cn_1d!(rhs, diag, off, line, cdt)
        line .= LU \ rhs
        for iz in 1:Nz; ψ[ix,iy,iz] = line[iz]; end
    end
end

# ════════════════════════════════════════════════════════════════
#  UN PASO COMPLETO (Strang + ADI alternante)
# ════════════════════════════════════════════════════════════════

function do_one_step!(R, P, ψ, V, g::Grid3D, sys::PhysSystem,
                      masses, rcut2, M_abs, kin::KineticOps,
                      LU_x, LU_y, LU_z, dt_step, step_parity)
    hdt = 0.5 * dt_step
    cdt = 1im * hdt
    Fn = zeros(2,3); Fe = zeros(2,3); Ft = zeros(2,3)

    ftot!(F, Rl) = begin
        nuc_forces!(Fn, Rl, sys)
        electron_forces!(Fe, ψ, g, Rl, sys, rcut2)
        @inbounds for i in 1:2, j in 1:3
            F[i,j] = Fn[i,j] + Fe[i,j]
        end
    end

    classical_step!(R, P, masses, hdt, Ft, ftot!)
    potential!(V, g, R, sys, rcut2)
    expV!(ψ, V, hdt)

    if isodd(step_parity)
        sweep_z!(ψ, LU_z, kin.Hz_diag, kin.Hz_off, cdt)
        sweep_y!(ψ, LU_y, kin.Hy_diag, kin.Hy_off, cdt)
        sweep_x!(ψ, LU_x, kin.Hx_diag, kin.Hx_off, cdt)
    else
        sweep_x!(ψ, LU_x, kin.Hx_diag, kin.Hx_off, cdt)
        sweep_y!(ψ, LU_y, kin.Hy_diag, kin.Hy_off, cdt)
        sweep_z!(ψ, LU_z, kin.Hz_diag, kin.Hz_off, cdt)
    end

    expV!(ψ, V, hdt)
    classical_step!(R, P, masses, hdt, Ft, ftot!)
    apply_mask!(ψ, M_abs)
end

# ════════════════════════════════════════════════════════════════
#  PASO TEMPORAL
# ════════════════════════════════════════════════════════════════

function compute_dt(Δx, Δy, Δz, v0)
    Δm = min(Δx, Δy, Δz)
    dt_c = USE_COURANT ? C_COURANT * 4.0 * Δm^2 : DT_MANUAL
    USE_VEL_LIMIT ? min(dt_c, C_VEL * Δm / max(abs(v0), 1e-12)) : dt_c
end

# ════════════════════════════════════════════════════════════════
#  DIAGNÓSTICOS DE ENERGÍA
#
#  Cantidades raw (integrales sin normalizar):
#    T_e, V_eN, E_elec, E_total
#  Cantidades normalizadas (valores esperados por unidad de probabilidad):
#    T_e_norm = T_e/‖ψ‖², V_eN_norm, E_elec_norm, E_total_norm
#
#  E_total_norm = K_T + K_P + V_nn + E_elec/‖ψ‖²  es el observable
#  END natural. E_total_raw puede variar de forma no monótona debido
#  a la no conmutación entre la máscara absorbente y el operador cinético.
# ════════════════════════════════════════════════════════════════

"""⟨T_e⟩ = -(1/2M) ∫ ψ* ∇²ψ d³r  (Laplaciano discreto 3pt)."""
function kinetic_energy_elec(ψ::Array{ComplexF64,3}, g::Grid3D, M_eff::Float64)
    Nx, Ny, Nz = size(ψ)
    coeff = -0.5 / M_eff
    idx2 = 1.0/g.Δx^2; idy2 = 1.0/g.Δy^2; idz2 = 1.0/g.Δz^2
    T_e = 0.0
    @inbounds for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
        ψc = ψ[ix,iy,iz]
        ψxm = ix > 1  ? ψ[ix-1,iy,iz] : zero(ComplexF64)
        ψxp = ix < Nx ? ψ[ix+1,iy,iz] : zero(ComplexF64)
        ψym = iy > 1  ? ψ[ix,iy-1,iz] : zero(ComplexF64)
        ψyp = iy < Ny ? ψ[ix,iy+1,iz] : zero(ComplexF64)
        ψzm = iz > 1  ? ψ[ix,iy,iz-1] : zero(ComplexF64)
        ψzp = iz < Nz ? ψ[ix,iy,iz+1] : zero(ComplexF64)
        lap = (ψxp - 2ψc + ψxm)*idx2 + (ψyp - 2ψc + ψym)*idy2 + (ψzp - 2ψc + ψzm)*idz2
        T_e += real(conj(ψc) * lap)
    end
    return coeff * T_e * g.dV
end

"""⟨V_eN⟩ = ∫ |ψ|² V d³r."""
function potential_energy_elec(ψ::Array{ComplexF64,3}, V::Array{Float64,3}, dV::Float64)
    E = 0.0
    @inbounds for I in CartesianIndices(ψ); E += abs2(ψ[I]) * V[I]; end
    return E * dV
end

"""Energías cinéticas nucleares."""
function nuclear_kinetic(P::Matrix{Float64}, sys::PhysSystem)
    K_T = (P[1,1]^2 + P[1,2]^2 + P[1,3]^2) / (2.0 * sys.M_target)
    K_P = (P[2,1]^2 + P[2,2]^2 + P[2,3]^2) / (2.0 * sys.M_proj)
    return K_T, K_P
end

"""V_nn = ZT·ZP / r₁₂."""
function nuclear_coulomb(R::Matrix{Float64}, sys::PhysSystem)
    dx = R[1,1]-R[2,1]; dy = R[1,2]-R[2,2]; dz = R[1,3]-R[2,3]
    r12 = sqrt(dx*dx + dy*dy + dz*dz)
    r12 < 1e-12 && return 0.0
    return sys.ZT * sys.ZP / r12
end

"""Todas las componentes de energía → NamedTuple (raw + normalizadas)."""
function compute_energies(ψ, V, g::Grid3D, R, P, sys::PhysSystem, rcut2)
    potential!(V, g, R, sys, rcut2)
    T_e  = kinetic_energy_elec(ψ, g, sys.M_eff)
    V_eN = potential_energy_elec(ψ, V, g.dV)
    K_T, K_P = nuclear_kinetic(P, sys)
    V_nn = nuclear_coulomb(R, sys)
    nrm  = sum(abs2, ψ) * g.dV
    E_el = T_e + V_eN

    if nrm > 1e-14
        T_e_n  = T_e / nrm
        V_eN_n = V_eN / nrm
        E_el_n = E_el / nrm
    else
        T_e_n = NaN; V_eN_n = NaN; E_el_n = NaN
    end

    return (K_T=K_T, K_P=K_P, V_nn=V_nn, T_e=T_e, V_eN=V_eN,
            E_elec=E_el, E_total=K_T+K_P+V_nn+E_el, norm=nrm,
            T_e_norm=T_e_n, V_eN_norm=V_eN_n, E_elec_norm=E_el_n,
            E_total_norm=K_T+K_P+V_nn+E_el_n)
end

# ════════════════════════════════════════════════════════════════
#  DENSIDAD AXIAL Y PARTICIÓN Ω
# ════════════════════════════════════════════════════════════════

function axis_density(ψ::Array{ComplexF64,3}, g::Grid3D, dir::Symbol)
    Nx, Ny, Nz = size(ψ)
    if dir == :z
        rho = zeros(Nz)
        @inbounds for iz in 1:Nz
            a = 0.0
            for ix in 1:Nx, iy in 1:Ny; a += abs2(ψ[ix,iy,iz]); end
            rho[iz] = a * g.Δx * g.Δy
        end
        return [g.Z[iz+1] for iz in 1:Nz], rho
    elseif dir == :x
        rho = zeros(Nx)
        @inbounds for ix in 1:Nx
            a = 0.0
            for iy in 1:Ny, iz in 1:Nz; a += abs2(ψ[ix,iy,iz]); end
            rho[ix] = a * g.Δy * g.Δz
        end
        return [g.X[ix+1] for ix in 1:Nx], rho
    else
        rho = zeros(Ny)
        @inbounds for iy in 1:Ny
            a = 0.0
            for ix in 1:Nx, iz in 1:Nz; a += abs2(ψ[ix,iy,iz]); end
            rho[iy] = a * g.Δx * g.Δz
        end
        return [g.Y[iy+1] for iy in 1:Ny], rho
    end
end

function trapz(x, y)
    n = length(x); n < 2 && return 0.0; s = 0.0
    @inbounds for i in 1:n-1; s += 0.5*(x[i+1]-x[i])*(y[i+1]+y[i]); end
    return s
end

function find_omega_valley(s, rho, sT, sP, shift)
    lo, hi = minmax(sT, sP)
    idxs = findall(x -> lo <= x <= hi, s)
    length(idxs) < 3 && return 0.5*(sT+sP) + shift
    return s[idxs[argmin(rho[idxs])]] + shift
end

function split_density_at_omega(s, rho, s0, tsign)
    idx = tsign > 0 ? findall(x -> x >= s0, s) : findall(x -> x <= s0, s)
    isempty(idx) && return (0.0, trapz(s, rho))
    Pc = trapz(s[idx], rho[idx])
    return (Pc, trapz(s, rho) - Pc)
end

# ════════════════════════════════════════════════════════════════
#  AJUSTE ASINTÓTICO DE LA NORMA
#  Ne(t) = β·exp(-α·t) + N∞  →  Pi = 1 - N∞
#  Ref: Cabrera-Trujillo et al., PRA 108, 012817 (2023)
# ════════════════════════════════════════════════════════════════

"""Ajuste asintótico. Retorna (exitoso, N∞, α, β, residuo)."""
function fit_norm_asymptote(t_data::Vector{Float64}, norm_data::Vector{Float64};
                            n_alpha::Int = 200, alpha_range = (1e-5, 0.5),
                            min_points::Int = 15, max_residual::Float64 = 1e-3,
                            min_ionization::Float64 = 0.005)
    n = length(t_data)
    n < min_points && return (false, NaN, NaN, NaN, NaN)
    (1.0 - norm_data[end]) < min_ionization && return (false, NaN, NaN, NaN, NaN)

    log_a_min = log10(alpha_range[1])
    log_a_max = log10(alpha_range[2])
    alphas = 10.0 .^ range(log_a_min, log_a_max; length = n_alpha)

    best_res = Inf
    best_alpha = NaN; best_beta = NaN; best_Ninf = NaN

    for α in alphas
        sum_e2 = 0.0; sum_e = 0.0; sum_Ne = 0.0; sum_N = 0.0
        @inbounds for i in 1:n
            ei = exp(-α * t_data[i])
            sum_e2 += ei * ei
            sum_e  += ei
            sum_Ne += norm_data[i] * ei
            sum_N  += norm_data[i]
        end
        det = sum_e2 * n - sum_e * sum_e
        abs(det) < 1e-30 && continue
        β    = (sum_Ne * n - sum_N * sum_e) / det
        Ninf = (sum_e2 * sum_N - sum_e * sum_Ne) / det

        β < 0    && continue
        Ninf < 0 && continue
        Ninf > 1 && continue

        res = 0.0
        @inbounds for i in 1:n
            di = norm_data[i] - β * exp(-α * t_data[i]) - Ninf
            res += di * di
        end
        res /= n

        if res < best_res
            best_res = res; best_alpha = α; best_beta = β; best_Ninf = Ninf
        end
    end

    isnan(best_Ninf) && return (false, NaN, NaN, NaN, NaN)
    best_res > max_residual && return (false, NaN, NaN, NaN, best_res)
    return (true, best_Ninf, best_alpha, best_beta, best_res)
end

# ════════════════════════════════════════════════════════════════
#  CARPETAS Y UTILIDADES
# ════════════════════════════════════════════════════════════════

function pick_process(ZP, mode)
    mode == :auto && (ZP > 0 ? (return(:capture,true,false)) :
                      ZP < 0 ? (return(:ionize,false,true)) :
                               (return(:both,true,true)))
    mode == :capture && return(:capture, true, false)
    mode == :ionize  && return(:ionize, false, true)
    return(:both, true, true)
end

function system_folder_name(sys::PhysSystem)
    ZT, ZP, mt, mp = sys.ZT, sys.ZP, sys.M_target, sys.M_proj; a = 5.0
    ZT==1 && ZP==1  && isapprox(mt,1836.15;atol=a) && isapprox(mp,1836.15;atol=a) && return "p + H"
    ZT==1 && ZP==-1 && isapprox(mt,1836.15;atol=a) && isapprox(mp,1836.15;atol=a) && return "p̄ + H"
    ZT==2 && ZP==1   && return "p + He⁺"
    ZT==2 && ZP==-1  && return "p̄ + He⁺"
    ZT==2 && ZP==2  && isapprox(mp,7294.3;atol=a)  && return "α + He⁺"
    ZT==1 && ZP==2  && isapprox(mp,7294.3;atol=a)  && return "α + H"
    return @sprintf("Z%g+Z%g", ZT, ZP)
end

ensure_dir(p) = (isdir(p) || mkpath(p); p)
energy_folder(E) = replace(@sprintf("%gkeV", E), "." => "p")
fmt_compact(x; ndigits=6) = replace(replace(@sprintf("%.*f", ndigits, x), r"0+$" => ""), r"\.$" => "")
b_folder(b) = "b = " * fmt_compact(b)

function create_b_dir(base, b)
    d = joinpath(base, b_folder(b))
    !isdir(d) && (mkpath(d); return d)
    k = 2
    while true
        c = d * "__$k"
        !isdir(c) && (mkpath(c); return c)
        k += 1
    end
end

@inline axis_index(dir::Symbol) = dir == :x ? 1 : dir == :y ? 2 : 3

function initial_conditions(b, E_keV, col_axis::Symbol, col_sign::Int,
                            b_axis::Symbol, sys::PhysSystem)
    p = momentum_from_keV(E_keV, sys.M_proj)
    R = zeros(2, 3); P = zeros(2, 3)
    R[1,:] .= R_TARGET_INIT

    ax_col = axis_index(col_axis)
    ax_b   = axis_index(b_axis)

    R[2, ax_b]   = R_TARGET_INIT[ax_b] + b
    R[2, ax_col] = S_TRAJ[1]
    P[2, ax_col] = col_sign * p

    return R, P, p
end

function write_info_txt(path, title, entries)
    open(path, "w") do io
        println(io, "LattEND.jl v$VERSION_STR  |  $title")
        println(io, "Generado: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "─"^60)
        for (k, v) in entries; println(io, k, ": ", v); end
    end
end

function fmt_time(sec)
    sec < 60   && return @sprintf("%.1fs", sec)
    sec < 3600 && return @sprintf("%dm %02ds", div(Int(floor(sec)),60), mod(Int(floor(sec)),60))
    h = div(Int(floor(sec)), 3600); m = div(mod(Int(floor(sec)), 3600), 60)
    return @sprintf("%dh %02dm", h, m)
end

# ════════════════════════════════════════════════════════════════
#  DRIVER: UNA SIMULACIÓN
#  Loop limpio: propagar hasta que sP pase S_TRAJ[2] o t > TMAX_HARD
# ════════════════════════════════════════════════════════════════

function run_one!(sim_dir, b, E_keV, col_axis::Symbol, col_sign::Int,
                  b_axis::Symbol, g::Grid3D, sys::PhysSystem,
                  kin::KineticOps, M_abs, dt_base)
    t_wall0 = time_ns()
    masses = [sys.M_target, sys.M_proj]
    R, P, p = initial_conditions(b, E_keV, col_axis, col_sign, b_axis, sys)
    ax = axis_index(col_axis)
    tsign = sign(S_TRAJ[2] - S_TRAJ[1]) * col_sign
    v0 = P[2, ax] / sys.M_proj
    T_classical = abs(S_TRAJ[2] - S_TRAJ[1]) / max(abs(v0), 1e-12)
    rcut2 = USE_GRID_RCUT ? (RCUT_FACTOR * min(g.Δx, g.Δy, g.Δz))^2 : 1e-24

    nmax = Int(ceil(TMAX_HARD / dt_base))

    ψ = Array{ComplexF64,3}(undef, g.Nx, g.Ny, g.Nz)
    V = Array{Float64,3}(undef, g.Nx, g.Ny, g.Nz)
    init_psi_1s!(ψ, g, R_TARGET_INIT, sys.ZT)
    normalize!(ψ, g.dV)

    # Energías iniciales
    Ei = compute_energies(ψ, V, g, R, P, sys, rcut2)
    KP_init = Ei.K_P

    # Prefactorizar LU (un solo dt para todo el run)
    cdt0 = 1im * (dt_base / 2)
    LUx = build_cn_lu(kin.Hx_diag, kin.Hx_off, cdt0)
    LUy = build_cn_lu(kin.Hy_diag, kin.Hy_off, cdt0)
    LUz = build_cn_lu(kin.Hz_diag, kin.Hz_off, cdt0)

    # Archivos de diagnóstico
    io_tr = WRITE_TRAJ ? open(joinpath(sim_dir, "traj_log.dat"), "w") : nothing
    io_tr !== nothing && println(io_tr, "# t  xT yT zT  xP yP zP")

    do_elog = WRITE_ENERGY_LOG && ENERGY_LOG_EVERY > 0
    io_el = nothing
    if do_elog
        io_el = open(joinpath(sim_dir, "energy_log.dat"), "w")
        println(io_el, "# t  K_T  K_P  V_nn  T_e  V_eN  E_elec  E_total  norm  T_e_norm  V_eN_norm  E_elec_norm  E_total_norm")
        @printf(io_el, "%12.6f  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %12.8f  %16.8e  %16.8e  %16.8e  %16.8e\n",
                0.0, Ei.K_T, Ei.K_P, Ei.V_nn, Ei.T_e, Ei.V_eN, Ei.E_elec, Ei.E_total, Ei.norm,
                Ei.T_e_norm, Ei.V_eN_norm, Ei.E_elec_norm, Ei.E_total_norm)
    end

    # Historial de norma para ajuste asintótico (cada ENERGY_LOG_EVERY pasos)
    t_hist    = Float64[]
    norm_hist = Float64[]
    t_closest = 0.0
    r12_min   = Inf

    t = 0.0; nsteps = 0; stop_tag = "TMAX"

    # ── Loop principal ──
    for k in 1:nmax
        nsteps = k

        # ¿Proyectil pasó S_TRAJ[2]?
        sP = R[2, ax]
        if tsign != 0 && tsign * (sP - S_TRAJ[2]) > 0
            stop_tag = "PASS"
            break
        end

        # Propagar un paso
        do_one_step!(R, P, ψ, V, g, sys, masses, rcut2, M_abs, kin,
                     LUx, LUy, LUz, dt_base, k)
        t += dt_base

        # Trayectoria
        if io_tr !== nothing
            @printf(io_tr, "%12.6f  % .6f % .6f % .6f  % .6f % .6f % .6f\n",
                    t, R[1,1], R[1,2], R[1,3], R[2,1], R[2,2], R[2,3])
        end

        # Diagnósticos de energía + historial de norma
        if do_elog && mod(k, ENERGY_LOG_EVERY) == 0
            Ek = compute_energies(ψ, V, g, R, P, sys, rcut2)
            @printf(io_el, "%12.6f  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %12.8f  %16.8e  %16.8e  %16.8e  %16.8e\n",
                    t, Ek.K_T, Ek.K_P, Ek.V_nn, Ek.T_e, Ek.V_eN, Ek.E_elec, Ek.E_total, Ek.norm,
                    Ek.T_e_norm, Ek.V_eN_norm, Ek.E_elec_norm, Ek.E_total_norm)
            push!(t_hist, t)
            push!(norm_hist, Ek.norm)
        end

        # r₁₂ mínimo y t_closest
        dx12 = R[2,1]-R[1,1]; dy12 = R[2,2]-R[1,2]; dz12 = R[2,3]-R[1,3]
        r12 = sqrt(dx12*dx12 + dy12*dy12 + dz12*dz12)
        if r12 < r12_min
            r12_min = r12
            t_closest = t
        end
    end

    io_tr !== nothing && close(io_tr)

    # ── Energías finales ──
    Ef = compute_energies(ψ, V, g, R, P, sys, rcut2)
    if do_elog
        @printf(io_el, "%12.6f  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %12.8f  %16.8e  %16.8e  %16.8e  %16.8e\n",
                t, Ef.K_T, Ef.K_P, Ef.V_nn, Ef.T_e, Ef.V_eN, Ef.E_elec, Ef.E_total, Ef.norm,
                Ef.T_e_norm, Ef.V_eN_norm, Ef.E_elec_norm, Ef.E_total_norm)
        close(io_el)
    end

    ΔE_proj   = Ef.K_P - KP_init
    KT_recoil = Ef.K_T
    ΔE_total  = Ef.E_total - Ei.E_total

    norm_f = Ef.norm
    s, rho = axis_density(ψ, g, col_axis)
    sT = R[1, ax]; sP_f = R[2, ax]
    s0 = OMEGA_MODE == :fixed ? OMEGA_FIXED :
         find_omega_valley(s, rho, sT, sP_f, OMEGA_SHIFT)
    Pcap, Ptar = split_density_at_omega(s, rho, s0, tsign)

    Pion_raw = max(1.0 - norm_f, 0.0)

    # ── Ajuste asintótico de la norma ──
    T_buffer = max(T_classical * 0.3, 10.0)
    t_fit_start = t_closest + T_buffer
    fit_idx = findall(ti -> ti >= t_fit_start, t_hist)

    fit_ok = false; Ninf_fit = NaN; alpha_fit = NaN; beta_fit = NaN; fit_res = NaN
    if length(fit_idx) >= 15
        fit_ok, Ninf_fit, alpha_fit, beta_fit, fit_res =
            fit_norm_asymptote(t_hist[fit_idx], norm_hist[fit_idx])
    end

    if fit_ok
        Pion = max(1.0 - Ninf_fit, 0.0)
        pion_method = "FIT"
        # Guarda: el ajuste no debe inflar Pion más allá de lo razonable
        if Pion > Pion_raw * 1.5 + 0.02
            Pion = Pion_raw
            pion_method = "RAW"
            fit_ok = false
        end
    else
        Pion = Pion_raw
        pion_method = "RAW"
    end

    if WRITE_DENSITY
        al = col_axis == :z ? "z" : col_axis == :x ? "x" : "y"
        open(joinpath(sim_dir, "dens_final_$(al).dat"), "w") do io
            println(io, "# s(a.u.)   rho_s")
            for i in eachindex(s)
                @printf(io, "%12.6f  %20.12e\n", s[i], rho[i])
            end
        end
    end

    wsec = (time_ns() - t_wall0) / 1e9

    dir_label = string(col_axis) * (col_sign < 0 ? "_neg" : "")

    entries = Pair{String,Any}[
        "dir" => dir_label, "E_keV" => E_keV, "b" => b,
        "S_TRAJ" => string(S_TRAJ),
        "T_classical" => T_classical, "dt_base" => dt_base,
        "nsteps" => nsteps, "nmax" => nmax,
        "t_final_au" => t, "stop_tag" => stop_tag,
        "sP_final" => sP_f, "sT_final" => sT, "r12_min" => r12_min,
        "Omega" => s0,
        "Pcap" => Pcap, "Ptar" => Ptar, "Pion" => Pion, "Pion_raw" => Pion_raw,
        "Pion_method" => pion_method, "norm_final" => norm_f,
        "─── Ajuste asintótico ───" => "",
        "Ninf_fit" => fit_ok ? Ninf_fit : "N/A",
        "alpha_fit" => fit_ok ? alpha_fit : "N/A",
        "beta_fit" => fit_ok ? beta_fit : "N/A",
        "fit_residual" => fit_ok ? fit_res : "N/A",
        "fit_window" => fit_ok ? @sprintf("[%.1f, %.1f] (%d pts)",
            t_hist[fit_idx[1]], t_hist[fit_idx[end]], length(fit_idx)) : "N/A",
        "─── Energías iniciales ───" => "",
        "K_T_init" => Ei.K_T, "K_P_init" => Ei.K_P, "V_nn_init" => Ei.V_nn,
        "T_e_init" => Ei.T_e, "V_eN_init" => Ei.V_eN,
        "E_elec_init" => Ei.E_elec, "E_total_init" => Ei.E_total,
        "─── Energías finales ───" => "",
        "K_T_final" => Ef.K_T, "K_P_final" => Ef.K_P, "V_nn_final" => Ef.V_nn,
        "T_e_final" => Ef.T_e, "V_eN_final" => Ef.V_eN,
        "E_elec_final" => Ef.E_elec, "E_total_final" => Ef.E_total,
        "─── Balance energético ───" => "",
        "ΔE_proj" => ΔE_proj, "K_T_recoil" => KT_recoil,
        "ΔE_total" => ΔE_total,
        "ΔE_total/|E_init|" => Ei.E_total != 0 ? ΔE_total/abs(Ei.E_total) : NaN,
        "wall_time_sec" => wsec,
    ]
    write_info_txt(joinpath(sim_dir, "run_info.txt"), "Run info", entries)

    return (Pcap=Pcap, Ptar=Ptar, Pion=Pion, Pion_raw=Pion_raw,
            norm_final=norm_f, s0=s0, t_final=t, nsteps=nsteps,
            sP_final=sP_f, sT_final=sT,
            stop_tag=stop_tag, wall_sec=wsec,
            T_classical=T_classical, dt_base=dt_base,
            ΔE_proj=ΔE_proj, K_T_recoil=KT_recoil,
            E_total_init=Ei.E_total, E_total_final=Ef.E_total,
            ΔE_total=ΔE_total, r12_min=r12_min, pion_method=pion_method)
end

# ════════════════════════════════════════════════════════════════
#  SUITE PRINCIPAL
# ════════════════════════════════════════════════════════════════

function main_suite()
    tw0 = time_ns()
    sys = PhysSystem(ZT, ZP, M_TARGET, M_PROJ, M_EFF)
    g = build_grid(XMIN, XMAX, NX, YMIN, YMAX, NY, ZMIN, ZMAX, NZ)
    kin = build_kinetic_ops(sys, g)
    Mab = build_absorbing_mask(g, LX_ABS, LY_ABS, LZ_ABS,
                               XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

    (tag, DO_CAP, DO_ION) = pick_process(sys.ZP, PROCESS_MODE)
    pf = DO_CAP && !DO_ION ? "Capture" :
         !DO_CAP && DO_ION ? "Ionization" : "CapIon"
    sn = system_folder_name(sys)
    root = ensure_dir(joinpath(BASE_PATH, sn, pf))
    rdir = ensure_dir(joinpath(root, "runset_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS")))

    println("═"^62)
    println("  LattEND.jl v$VERSION_STR — Lattice END 3D")
    println("  Sistema: $sn (ZT=$(sys.ZT), ZP=$(sys.ZP))")
    println("  Modo: $tag")
    !isempty(USER_NOTE) && println("  Nota: $USER_NOTE")
    println("  Carpeta: $rdir")
    println("═"^62)
    @printf("  Dominio: [%.1f,%.1f]×[%.1f,%.1f]×[%.1f,%.1f] a.u.\n",
            XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
    @printf("  Malla: %d×%d×%d  (Δ=%.4f, %.4f, %.4f)\n",
            g.Nx, g.Ny, g.Nz, g.Δx, g.Δy, g.Δz)
    @printf("  Absorbente: Lx=%.1f, Ly=%.1f, Lz=%.1f\n", LX_ABS, LY_ABS, LZ_ABS)
    @printf("  Masas: M_T=%.6f, M_P=%.6f m_e\n", sys.M_target, sys.M_proj)
    @printf("  S_TRAJ: [%.1f → %.1f]  TMAX=%.0f a.u.\n", S_TRAJ[1], S_TRAJ[2], TMAX_HARD)
    @printf("  B_DIR: %s\n", string(B_DIR))
    println("  ADI: alternante (z-y-x / x-y-z)")
    elog_msg = ENERGY_LOG_EVERY > 0 ? "cada $(ENERGY_LOG_EVERY) pasos" : "desactivado"
    println("  Energía: diagnósticos ON (log $elog_msg)")
    println("─"^62)

    sp = joinpath(rdir, "summary.dat")
    open(sp, "w") do io
        println(io, "# LattEND.jl v$VERSION_STR — Summary")
        println(io, "# dir  E(keV)  b  Pcap  Ptar  Pion  norm  bPcap  bPion  Omega  t_final  nsteps  ΔE_proj  K_T_recoil  ΔE_total  wall_sec  stop  sim_folder")
    end

    tr = 0; tw = 0.0; tp = 0.0

    for (col_axis, col_sign) in DIRECTIONS
        dir_label = string(col_axis) * (col_sign < 0 ? "_neg" : "")
        dd = ensure_dir(joinpath(rdir, "dir_" * dir_label))

        for Ek in E_KEV_LIST
            Ed = ensure_dir(joinpath(dd, energy_folder(Ek)))
            p0 = momentum_from_keV(Ek, sys.M_proj)
            v0 = p0 / sys.M_proj
            dt = compute_dt(g.Δx, g.Δy, g.Δz, v0)
            ns = length(B_LIST)
            T_cl = abs(S_TRAJ[2] - S_TRAJ[1]) / max(abs(v0), 1e-12)

            @printf("\n▶ dir=%s  E=%.1f keV  Δt=%.6f a.u.  T_cl=%.1f  (%d sims)\n",
                    dir_label, Ek, dt, T_cl, ns)
            @printf("  %-7s %-10s %-10s %-10s %-10s %-9s %-6s %-8s %-8s %-8s %-8s\n",
                    "b", "Pcap", "Pion", "norm", "sP", "ΔE_proj", "stop", "t_s", "t_a", "t_prom", "t_est")

            # observables.dat — 19 columnas (col 19 = Pion_raw)
            obs_path = joinpath(Ed, "observables.dat")
            open(obs_path, "w") do io
                println(io, "# b  Pcap  Ptar  Pion  norm  bPcap  bPion  Omega  t_final  nsteps  sP_final  sT_final  r12_min  ΔE_proj  K_T_recoil  ΔE_total  stop  wall_sec  Pion_raw")
            end

            bw = 0.0
            block_results = []

            for (ib, b) in enumerate(B_LIST)
                sd = create_b_dir(Ed, b); snm = basename(sd)
                res = run_one!(sd, b, Ek, col_axis, col_sign, B_DIR,
                               g, sys, kin, Mab, dt)
                push!(block_results, res)
                tr += 1; tw += res.wall_sec; tp += res.t_final; bw += res.wall_sec
                tpr = bw / ib; rest = ns - ib; test = tpr * rest

                # Indicador de método: • = FIT
                pion_str = @sprintf("%.6f%s", res.Pion,
                                    res.pion_method == "FIT" ? "•" : "")

                @printf("  %-7s %-10s %-10s %-10s %-10s %-9s %-6s %-8s %-8s %-8s %-8s\n",
                        fmt_compact(b; ndigits=3),
                        @sprintf("%.6f", res.Pcap), pion_str,
                        @sprintf("%.6f", res.norm_final),
                        @sprintf("%.2f", res.sP_final),
                        @sprintf("%.4f", res.ΔE_proj), res.stop_tag,
                        fmt_time(res.wall_sec), fmt_time(bw), fmt_time(tpr),
                        rest > 0 ? fmt_time(test) : "—")

                # observables.dat — 19 columnas
                open(obs_path, "a") do io
                    @printf(io, "%10.6f  %12.6e  %12.6e  %12.6e  %12.8f  %12.6e  %12.6e  %10.6f  %10.4f  %6d  %10.6f  %10.6f  %10.6f  %16.8e  %16.8e  %16.8e  %s  %10.4f  %12.6e\n",
                            b, res.Pcap, res.Ptar, res.Pion, res.norm_final,
                            b*res.Pcap, b*res.Pion, res.s0, res.t_final, res.nsteps,
                            res.sP_final, res.sT_final, res.r12_min,
                            res.ΔE_proj, res.K_T_recoil, res.ΔE_total,
                            res.stop_tag, res.wall_sec, res.Pion_raw)
                end

                # Summary global
                open(sp, "a") do io
                    @printf(io, "%s  %10.6f  %10.6f  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %12.6e  %10.6f  %10.6f  %8d  %16.8e  %16.8e  %16.8e  %10.4f  %s  %s/%s/%s\n",
                            dir_label, Ek, b,
                            res.Pcap, res.Ptar, res.Pion, res.norm_final,
                            b*res.Pcap, b*res.Pion, res.s0,
                            res.t_final, res.nsteps,
                            res.ΔE_proj, res.K_T_recoil, res.ΔE_total,
                            res.wall_sec, res.stop_tag,
                            "dir_"*dir_label, energy_folder(Ek), snm)
                end
            end

            # Resumen de bloque
            avg_Pcap = ns > 0 ? sum(r -> r.Pcap, block_results)/ns : 0.0
            avg_norm = ns > 0 ? sum(r -> r.norm_final, block_results)/ns : 0.0
            avg_Pion = ns > 0 ? sum(r -> r.Pion, block_results)/ns : 0.0
            avg_dE   = ns > 0 ? sum(r -> r.ΔE_proj, block_results)/ns : 0.0

            @printf("  Resumen E=%.3f keV: %d sims, t=%s\n", Ek, ns, fmt_time(bw))
            @printf("    ⟨Pcap⟩=%.6f  ⟨Pion⟩=%.6f  ⟨norm⟩=%.6f  ⟨ΔE⟩=%.4f\n",
                    avg_Pcap, avg_Pion, avg_norm, avg_dE)

            open(joinpath(Ed, "block_summary.txt"), "w") do io
                println(io, "LattEND.jl v$VERSION_STR  |  E=$(Ek) keV, dir=$dir_label")
                println(io, "─"^50)
                @printf(io, "n_sims:     %d\n", ns)
                @printf(io, "wall_time:  %s\n", fmt_time(bw))
                @printf(io, "⟨Pcap⟩:     %.8f\n", avg_Pcap)
                @printf(io, "⟨Pion⟩:     %.8f\n", avg_Pion)
                @printf(io, "⟨norm⟩:     %.8f\n", avg_norm)
                @printf(io, "⟨ΔE_proj⟩:  %.8f a.u.\n", avg_dE)
            end
            println("─"^62)
        end
    end

    wt = (time_ns() - tw0) / 1e9
    avg_wall = tr > 0 ? wt/tr : 0.0

    eg = Pair{String,Any}[
        "code" => "LattEND.jl v$VERSION_STR",
        "system" => sn, "process" => pf,
        "runset_dir" => rdir, "note" => USER_NOTE,
        "ZT" => sys.ZT, "ZP" => sys.ZP,
        "M_target" => sys.M_target, "M_proj" => sys.M_proj, "M_eff" => sys.M_eff,
        "R_target_init" => string(R_TARGET_INIT),
        "initial_wf" => "1s: (Z^{3/2}/√π)exp(-Z|r-R_T|)",
        "directions" => string(DIRECTIONS), "B_DIR" => string(B_DIR),
        "b_list" => string(B_LIST), "E_list" => string(E_KEV_LIST),
        "domain_x" => @sprintf("[%.2f,%.2f] Nx=%d", XMIN, XMAX, NX),
        "domain_y" => @sprintf("[%.2f,%.2f] Ny=%d", YMIN, YMAX, NY),
        "domain_z" => @sprintf("[%.2f,%.2f] Nz=%d", ZMIN, ZMAX, NZ),
        "Δ" => @sprintf("%.6f, %.6f, %.6f", g.Δx, g.Δy, g.Δz),
        "mask" => "cos^(1/8)",
        "L_abs" => @sprintf("%.1f, %.1f, %.1f", LX_ABS, LY_ABS, LZ_ABS),
        "dt_mode" => @sprintf("Cou=%s(C=%.2f) vel=%s(Cv=%.2f)",
                              USE_COURANT, C_COURANT, USE_VEL_LIMIT, C_VEL),
        "ADI" => "alternante z-y-x / x-y-z",
        "energy_log" => @sprintf("cada %d pasos", ENERGY_LOG_EVERY),
        "S_TRAJ" => string(S_TRAJ), "TMAX_HARD" => TMAX_HARD,
        "OMEGA" => "$(OMEGA_MODE) (fixed=$(OMEGA_FIXED))",
        "RCUT" => USE_GRID_RCUT ? @sprintf("grid-scaled (%.2f)", RCUT_FACTOR) : "off (1e-24)",
        "total_runs" => tr, "total_phys_time" => tp,
        "wall_time" => @sprintf("%.2f s (%s)", wt, fmt_time(wt)),
    ]
    write_info_txt(joinpath(rdir, "runset_info.txt"), "Global runset info", eg)

    println()
    println("═"^62)
    println("  RESUMEN FINAL — LattEND.jl v$VERSION_STR")
    @printf("  Simulaciones:  %d\n", tr)
    @printf("  Tiempo total:  %s\n", fmt_time(wt))
    @printf("  Tiempo/sim:    %s\n", fmt_time(avg_wall))
    println("  Summary:  $sp")
    println("  Carpeta:  $rdir")
    println("═"^62)
end

# ════════════════════════════════════════════════════════════════
const THIS_FILE = abspath(@__FILE__)
if (abspath(PROGRAM_FILE) == THIS_FILE) || isinteractive()
    Base.invokelatest(main_suite)
end
