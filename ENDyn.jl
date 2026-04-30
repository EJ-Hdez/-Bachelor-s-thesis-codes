# ════════════════════════════════════════════════════════════════
# ENDyn.jl  v2.2.0
# Electron-Nuclear Dynamics — Heavy data collector
#
# Esquema de propagación temporal (un paso completo Δt):
#   1. Nuclear VV (Δt/2)
#   2. exp(-i V Δt/2)
#   3. CN-ADI alternante (impares: z→y→x, pares: x→y→z)
#   4. exp(-i V Δt/2)
#   5. Nuclear VV (Δt/2)
#   6. Máscara absorbente cos^(1/8)
#
# Historial:
#   v1.0.0 — Motor físico prototipo
#   v2.0.0 — Barrido E×b, ADI alternante, diagnósticos de energía
#   v2.1.0 — Pcap/Pion, centro de masa, ángulo de dispersión
#   v2.2.0 — Progreso en terminal cada N pasos, ángulo de dispersión
#            con fórmula de momentos (Cabrera-Trujillo), estructura
#            de carpetas por sistema/energía/b, unidades eV/keV
# ════════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Dates
using FFTW

const CODE_NAME    = "ENDyn.jl"
const CODE_VERSION = "2.2.0"

# ════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DEL USUARIO
# ════════════════════════════════════════════════════════════════

const USER_NOTE = "Colisión p̄ + He⁺"
const BASE_PATH = "/media/user/your/path/here"

# ── Sistema físico ─────────────────────────────────────────────
const ZT       = 2.0
const ZP       = -1.0
const M_TARGET = 7294.299665
const M_PROJ   = 1836.152673426
const M_EFF    = 1.0

# ── Dominio y malla ───────────────────────────────────────────
const XMIN, XMAX, NX = -15.1, 15.1, 150
const YMIN, YMAX, NY = -15.1, 15.1, 150
const ZMIN, ZMAX, NZ = -18.1, 18.1, 180

# ── Máscara absorbente ────────────────────────────────────────
const LX_ABS = 4.0
const LY_ABS = 4.0
const LZ_ABS = 4.0

# ── Paso temporal ─────────────────────────────────────────────
const USE_COURANT   = true
const C_COURANT     = 0.50
const DT_MANUAL     = 0.08
const USE_VEL_LIMIT = true
const C_VEL         = 0.25

# ── Geometría de la colisión ──────────────────────────────────
const DIR_COLLISION = :z
const DIR_B         = :y
const R_TARGET_INIT = [0.0, 0.0, 0.0]
const S_TRAJ        = [-8.0, 12.0]   # [posición inicial, posición final]

# ── Barrido ───────────────────────────────────────────────────
const E_KEV_LIST = [0.01]
const B_LIST     = [0.22]

# ── Selección de proceso ──────────────────────────────────────
const PROCESS_MODE = :ionize    # :auto, :capture, :ionize, :both
const OMEGA        = 0.0     # plano de partición para captura (a.u.)

# ── Opciones de salida ────────────────────────────────────────
const WRITE_TRAJ_LOG    = true
const WRITE_ENERGY_LOG  = true
const ENERGY_LOG_EVERY  = 5
const WRITE_SNAPSHOTS   = true
const SNAP_EVERY        = 12.5

# ── Progreso en terminal ──────────────────────────────────────
const PROGRESS_EVERY = 100    # imprimir progreso cada N pasos

# ════════════════════════════════════════════════════════════════
#  CONSTANTES Y TIPOS
# ════════════════════════════════════════════════════════════════

const HARTREE_TO_EV = 27.211386245988

struct PhysSystem
    ZT::Float64
    ZP::Float64
    M_target::Float64
    M_proj::Float64
    M_eff::Float64
end

struct Grid3D
    X::Vector{Float64}
    Y::Vector{Float64}
    Z::Vector{Float64}
    Nx::Int
    Ny::Int
    Nz::Int
    Δx::Float64
    Δy::Float64
    Δz::Float64
    dV::Float64
end

struct KineticOps
    Hx_d::Vector{Float64}
    Hx_o::Vector{Float64}
    Hy_d::Vector{Float64}
    Hy_o::Vector{Float64}
    Hz_d::Vector{Float64}
    Hz_o::Vector{Float64}
end

# ════════════════════════════════════════════════════════════════
#  UTILIDADES GENERALES
# ════════════════════════════════════════════════════════════════

function momentum_from_keV(E_keV::Float64, M::Float64)
    E_Ha = (E_keV * 1000.0) / HARTREE_TO_EV
    return sqrt(2.0 * M * E_Ha)
end

function ax_idx(dir::Symbol)
    dir == :x && return 1
    dir == :y && return 2
    return 3
end

function fmt_time(s::Float64)
    if s < 60
        return @sprintf("%.1fs", s)
    elseif s < 3600
        m = div(Int(floor(s)), 60)
        sec = mod(Int(floor(s)), 60)
        return @sprintf("%dm%02ds", m, sec)
    else
        h = div(Int(floor(s)), 3600)
        m = div(mod(Int(floor(s)), 3600), 60)
        return @sprintf("%dh%02dm", h, m)
    end
end

function ensure_dir(p::String)
    isdir(p) || mkpath(p)
    return p
end

function center_of_mass(R::Matrix{Float64}, sys::PhysSystem)
    Mt = sys.M_target + sys.M_proj
    return ((sys.M_target * R[1,1] + sys.M_proj * R[2,1]) / Mt,
            (sys.M_target * R[1,2] + sys.M_proj * R[2,2]) / Mt,
            (sys.M_target * R[1,3] + sys.M_proj * R[2,3]) / Mt)
end

# ════════════════════════════════════════════════════════════════
#  SISTEMA FÍSICO: ETIQUETA Y CARPETAS
#
#  Estructura: BASE_PATH / system_label / Colisiones /
#              energy_folder / b_folder / simulacion_timestamp
# ════════════════════════════════════════════════════════════════

function system_label(sys::PhysSystem)
    mp = 1836.152673426   # protón
    ma = 7294.299665      # partícula alpha
    tol = 0.01            # tolerancia para masas precisas

    if sys.ZT == 1 && sys.ZP == 1 && isapprox(sys.M_target, mp; atol=tol) && isapprox(sys.M_proj, mp; atol=tol)
        return "p + H"
    elseif sys.ZT == 1 && sys.ZP == -1 && isapprox(sys.M_target, mp; atol=tol) && isapprox(sys.M_proj, mp; atol=tol)
        return "p̄ + H"
    elseif sys.ZT == 2 && sys.ZP == 1 && isapprox(sys.M_proj, mp; atol=tol)
        return "p + He⁺"
    elseif sys.ZT == 2 && sys.ZP == -1 && isapprox(sys.M_proj, mp; atol=tol)
        return "p̄ + He⁺"
    elseif sys.ZT == 1 && sys.ZP == 2 && isapprox(sys.M_proj, ma; atol=tol)
        return "α + H"
    elseif sys.ZT == 2 && sys.ZP == 2 && isapprox(sys.M_proj, ma; atol=tol)
        return "α + He⁺"
    else
        return @sprintf("Z%g + Z%g", sys.ZT, sys.ZP)
    end
end

"""Nombre de carpeta por energía: keV si E≥1, eV si E<1."""
function energy_folder_name(E_keV::Float64)
    if E_keV >= 1.0
        return @sprintf("%.1f keV", E_keV)
    else
        E_eV = E_keV * 1000.0
        if E_eV == floor(E_eV)
            return @sprintf("%d eV", Int(E_eV))
        else
            return @sprintf("%.1f eV", E_eV)
        end
    end
end

function b_folder_name(b::Float64)
    return @sprintf("b = %.1f", b)
end

"""Construye la ruta completa de simulación:
   BASE_PATH / system / Colisiones / energía / b / simulacion_timestamp"""
function build_sim_path(sys::PhysSystem, E_keV::Float64, b::Float64)
    sn = system_label(sys)
    sys_dir = ensure_dir(joinpath(BASE_PATH, sn))
    anim_dir = ensure_dir(joinpath(sys_dir, "Colisiones"))
    e_dir = ensure_dir(joinpath(anim_dir, energy_folder_name(E_keV)))
    b_dir = ensure_dir(joinpath(e_dir, b_folder_name(b)))
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    sim_dir = ensure_dir(joinpath(b_dir, "simulacion_" * timestamp))
    return sim_dir, e_dir
end

# ════════════════════════════════════════════════════════════════
#  MALLA Y HAMILTONIANOS
# ════════════════════════════════════════════════════════════════

function build_grid(xm, xM, Nx, ym, yM, Ny, zm, zM, Nz)
    Δx = (xM - xm) / (Nx + 1)
    Δy = (yM - ym) / (Ny + 1)
    Δz = (zM - zm) / (Nz + 1)
    X = [xm + i * Δx for i in 0:Nx+1]
    Y = [ym + j * Δy for j in 0:Ny+1]
    Z = [zm + k * Δz for k in 0:Nz+1]
    return Grid3D(X, Y, Z, Nx, Ny, Nz, Δx, Δy, Δz, Δx * Δy * Δz)
end

function build_kinetic_ops(sys::PhysSystem, g::Grid3D)
    m = 0.5 / sys.M_eff
    return KineticOps(
        fill(2m / g.Δx^2, g.Nx), fill(-m / g.Δx^2, g.Nx - 1),
        fill(2m / g.Δy^2, g.Ny), fill(-m / g.Δy^2, g.Ny - 1),
        fill(2m / g.Δz^2, g.Nz), fill(-m / g.Δz^2, g.Nz - 1))
end

# ════════════════════════════════════════════════════════════════
#  MÁSCARA ABSORBENTE cos^(1/8)
# ════════════════════════════════════════════════════════════════

function build_1d_mask(coords::Vector{Float64}, N::Int,
                       umin::Float64, umax::Float64, Labs::Float64)
    mask = ones(Float64, N)
    Labs <= 0 && return mask
    L = min(Labs, (umax - umin) / 2)
    @inbounds for i in 1:N
        u = coords[i+1]; m = 1.0
        if abs(umin - u) < L
            m *= abs(cos(π * abs(u - umin + L) / (2L)))^(1 / 8)
        end
        if abs(umax - u) < L
            m *= abs(cos(π * abs(u - umax + L) / (2L)))^(1 / 8)
        end
        mask[i] = m
    end
    return mask
end

function build_absorbing_mask(g::Grid3D, Lx, Ly, Lz)
    mx = build_1d_mask(g.X, g.Nx, Float64(XMIN), Float64(XMAX), Lx)
    my = build_1d_mask(g.Y, g.Ny, Float64(YMIN), Float64(YMAX), Ly)
    mz = build_1d_mask(g.Z, g.Nz, Float64(ZMIN), Float64(ZMAX), Lz)
    M = Array{Float64,3}(undef, g.Nx, g.Ny, g.Nz)
    @inbounds for ix in 1:g.Nx, iy in 1:g.Ny, iz in 1:g.Nz
        M[ix, iy, iz] = mx[ix] * my[iy] * mz[iz]
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
                      RT::Vector{Float64}, ZT_::Float64)
    pref = ZT_^(3 / 2) / sqrt(π)
    @inbounds for ix in 1:g.Nx, iy in 1:g.Ny, iz in 1:g.Nz
        r = sqrt((g.X[ix+1] - RT[1])^2 + (g.Y[iy+1] - RT[2])^2 + (g.Z[iz+1] - RT[3])^2)
        ψ[ix, iy, iz] = pref * exp(-ZT_ * r) + 0im
    end
end

function normalize!(ψ::Array{ComplexF64,3}, dV::Float64)
    ψ ./= sqrt(sum(abs2, ψ) * dV)
end

# ════════════════════════════════════════════════════════════════
#  POTENCIAL Y FUERZAS
# ════════════════════════════════════════════════════════════════

function potential!(V::Array{Float64,3}, g::Grid3D,
                    R::Matrix{Float64}, sys::PhysSystem)
    @inbounds for ix in 1:g.Nx, iy in 1:g.Ny, iz in 1:g.Nz
        x = g.X[ix+1]; y = g.Y[iy+1]; z = g.Z[iz+1]
        rT = sqrt((x - R[1,1])^2 + (y - R[1,2])^2 + (z - R[1,3])^2)
        rP = sqrt((x - R[2,1])^2 + (y - R[2,2])^2 + (z - R[2,3])^2)
        V[ix, iy, iz] = -sys.ZT / rT - sys.ZP / rP
    end
end

function nuc_forces!(F::Matrix{Float64}, R::Matrix{Float64}, sys::PhysSystem)
    dx = R[1,1] - R[2,1]; dy = R[1,2] - R[2,2]; dz = R[1,3] - R[2,3]
    r2 = max(dx^2 + dy^2 + dz^2, 1e-24); r = sqrt(r2)
    c = sys.ZT * sys.ZP / (r2 * r)
    F[1,1] = c*dx;  F[1,2] = c*dy;  F[1,3] = c*dz
    F[2,1] = -c*dx; F[2,2] = -c*dy; F[2,3] = -c*dz
end

function electron_forces!(Fe::Matrix{Float64}, ψ::Array{ComplexF64,3},
                          g::Grid3D, R::Matrix{Float64}, sys::PhysSystem)
    Fe .= 0.0
    rc2 = 1e-4
    @inbounds for ix in 1:g.Nx, iy in 1:g.Ny, iz in 1:g.Nz
        ρ = abs2(ψ[ix, iy, iz])
        ρ == 0.0 && continue
        x = g.X[ix+1]; y = g.Y[iy+1]; z = g.Z[iz+1]

        rxT = x - R[1,1]; ryT = y - R[1,2]; rzT = z - R[1,3]
        rT2 = max(rxT^2 + ryT^2 + rzT^2, rc2)
        iT3 = 1.0 / (rT2 * sqrt(rT2))
        Fe[1,1] += sys.ZT * ρ * rxT * iT3
        Fe[1,2] += sys.ZT * ρ * ryT * iT3
        Fe[1,3] += sys.ZT * ρ * rzT * iT3

        rxP = x - R[2,1]; ryP = y - R[2,2]; rzP = z - R[2,3]
        rP2 = max(rxP^2 + ryP^2 + rzP^2, rc2)
        iP3 = 1.0 / (rP2 * sqrt(rP2))
        Fe[2,1] += sys.ZP * ρ * rxP * iP3
        Fe[2,2] += sys.ZP * ρ * ryP * iP3
        Fe[2,3] += sys.ZP * ρ * rzP * iP3
    end
    Fe .*= g.dV
end

# ════════════════════════════════════════════════════════════════
#  VELOCITY-VERLET + CN-ADI
# ════════════════════════════════════════════════════════════════

function classical_step!(R, P, masses, τ, Fb, force_fn!)
    force_fn!(Fb, R)
    @inbounds for i in 1:2, j in 1:3; P[i,j] += 0.5 * τ * Fb[i,j]; end
    @inbounds for i in 1:2, j in 1:3; R[i,j] += τ * P[i,j] / masses[i]; end
    force_fn!(Fb, R)
    @inbounds for i in 1:2, j in 1:3; P[i,j] += 0.5 * τ * Fb[i,j]; end
end

function expV!(ψ::Array{ComplexF64,3}, V::Array{Float64,3}, hdt::Float64)
    @inbounds for I in CartesianIndices(ψ)
        ψ[I] *= cis(-hdt * V[I])
    end
end

function rhs_cn!(dest, dia, off, ψl, cdt)
    N = length(dia)
    @inbounds begin
        dest[1] = dia[1] * ψl[1] + off[1] * ψl[2]
        for i in 2:N-1
            dest[i] = off[i-1] * ψl[i-1] + dia[i] * ψl[i] + off[i] * ψl[i+1]
        end
        dest[N] = off[N-1] * ψl[N-1] + dia[N] * ψl[N]
        @. dest = ψl - cdt * dest
    end
end

function build_cn_lu(d, o, cdt)
    N = length(d)
    return lu(Tridiagonal(cdt .* o, ones(ComplexF64, N) .+ cdt .* d, cdt .* o))
end

function sweep_x!(ψ, LU, d, o, c)
    Nx, Ny, Nz = size(ψ)
    rhs = Vector{ComplexF64}(undef, Nx); line = similar(rhs)
    @inbounds for iz in 1:Nz, iy in 1:Ny
        for ix in 1:Nx; line[ix] = ψ[ix, iy, iz]; end
        rhs_cn!(rhs, d, o, line, c); line .= LU \ rhs
        for ix in 1:Nx; ψ[ix, iy, iz] = line[ix]; end
    end
end

function sweep_y!(ψ, LU, d, o, c)
    Nx, Ny, Nz = size(ψ)
    rhs = Vector{ComplexF64}(undef, Ny); line = similar(rhs)
    @inbounds for iz in 1:Nz, ix in 1:Nx
        for iy in 1:Ny; line[iy] = ψ[ix, iy, iz]; end
        rhs_cn!(rhs, d, o, line, c); line .= LU \ rhs
        for iy in 1:Ny; ψ[ix, iy, iz] = line[iy]; end
    end
end

function sweep_z!(ψ, LU, d, o, c)
    Nx, Ny, Nz = size(ψ)
    rhs = Vector{ComplexF64}(undef, Nz); line = similar(rhs)
    @inbounds for iy in 1:Ny, ix in 1:Nx
        for iz in 1:Nz; line[iz] = ψ[ix, iy, iz]; end
        rhs_cn!(rhs, d, o, line, c); line .= LU \ rhs
        for iz in 1:Nz; ψ[ix, iy, iz] = line[iz]; end
    end
end

function do_one_step!(R, P, ψ, V, g, sys, masses, Mab, kin,
                      LUx, LUy, LUz, dt, step_k)
    hdt = 0.5 * dt; cdt = 1im * hdt
    Fn = zeros(2, 3); Fe = zeros(2, 3); Ft = zeros(2, 3)
    function ftot!(F, Rl)
        nuc_forces!(Fn, Rl, sys)
        electron_forces!(Fe, ψ, g, Rl, sys)
        @inbounds for i in 1:2, j in 1:3
            F[i,j] = Fn[i,j] + Fe[i,j]
        end
    end
    classical_step!(R, P, masses, hdt, Ft, ftot!)
    potential!(V, g, R, sys)
    expV!(ψ, V, hdt)
    if isodd(step_k)
        sweep_z!(ψ, LUz, kin.Hz_d, kin.Hz_o, cdt)
        sweep_y!(ψ, LUy, kin.Hy_d, kin.Hy_o, cdt)
        sweep_x!(ψ, LUx, kin.Hx_d, kin.Hx_o, cdt)
    else
        sweep_x!(ψ, LUx, kin.Hx_d, kin.Hx_o, cdt)
        sweep_y!(ψ, LUy, kin.Hy_d, kin.Hy_o, cdt)
        sweep_z!(ψ, LUz, kin.Hz_d, kin.Hz_o, cdt)
    end
    expV!(ψ, V, hdt)
    classical_step!(R, P, masses, hdt, Ft, ftot!)
    apply_mask!(ψ, Mab)
end

function compute_dt(Δx, Δy, Δz, v0)
    Δm = min(Δx, Δy, Δz)
    dtc = USE_COURANT ? C_COURANT * 4.0 * Δm^2 : DT_MANUAL
    if USE_VEL_LIMIT
        dtv = C_VEL * Δm / max(abs(v0), 1e-12)
        return min(dtc, dtv, DT_MANUAL)
    end
    return min(dtc, DT_MANUAL)
end

# ════════════════════════════════════════════════════════════════
#  DIAGNÓSTICOS DE ENERGÍA (raw + normalizados)
# ════════════════════════════════════════════════════════════════

function kinetic_energy_elec(ψ::Array{ComplexF64,3}, g::Grid3D, Me::Float64)
    Nx, Ny, Nz = size(ψ)
    coeff = -0.5 / Me
    idx2 = 1.0 / g.Δx^2; idy2 = 1.0 / g.Δy^2; idz2 = 1.0 / g.Δz^2
    Te = 0.0
    @inbounds for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
        ψc = ψ[ix, iy, iz]
        ψxm = ix > 1  ? ψ[ix-1, iy, iz] : zero(ComplexF64)
        ψxp = ix < Nx ? ψ[ix+1, iy, iz] : zero(ComplexF64)
        ψym = iy > 1  ? ψ[ix, iy-1, iz] : zero(ComplexF64)
        ψyp = iy < Ny ? ψ[ix, iy+1, iz] : zero(ComplexF64)
        ψzm = iz > 1  ? ψ[ix, iy, iz-1] : zero(ComplexF64)
        ψzp = iz < Nz ? ψ[ix, iy, iz+1] : zero(ComplexF64)
        lap = (ψxp - 2ψc + ψxm) * idx2 +
              (ψyp - 2ψc + ψym) * idy2 +
              (ψzp - 2ψc + ψzm) * idz2
        Te += real(conj(ψc) * lap)
    end
    return coeff * Te * g.dV
end

function potential_energy_elec(ψ::Array{ComplexF64,3}, V::Array{Float64,3}, dV::Float64)
    E = 0.0
    @inbounds for I in CartesianIndices(ψ)
        E += abs2(ψ[I]) * V[I]
    end
    return E * dV
end

function compute_energies(ψ, V, g, R, P, sys)
    potential!(V, g, R, sys)
    Te = kinetic_energy_elec(ψ, g, sys.M_eff)
    VeN = potential_energy_elec(ψ, V, g.dV)
    KT = (P[1,1]^2 + P[1,2]^2 + P[1,3]^2) / (2.0 * sys.M_target)
    KP = (P[2,1]^2 + P[2,2]^2 + P[2,3]^2) / (2.0 * sys.M_proj)
    dx = R[1,1] - R[2,1]; dy = R[1,2] - R[2,2]; dz = R[1,3] - R[2,3]
    r12 = sqrt(dx^2 + dy^2 + dz^2)
    Vnn = r12 < 1e-12 ? 0.0 : sys.ZT * sys.ZP / r12
    nrm = sum(abs2, ψ) * g.dV
    Eel = Te + VeN
    Ten  = nrm > 1e-14 ? Te / nrm  : NaN
    VeNn = nrm > 1e-14 ? VeN / nrm : NaN
    Eeln = nrm > 1e-14 ? Eel / nrm : NaN
    return (K_T=KT, K_P=KP, V_nn=Vnn, T_e=Te, V_eN=VeN,
            E_elec=Eel, E_total=KT + KP + Vnn + Eel, norm=nrm,
            T_e_norm=Ten, V_eN_norm=VeNn, E_elec_norm=Eeln,
            E_total_norm=KT + KP + Vnn + Eeln)
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
            for ix in 1:Nx, iy in 1:Ny; a += abs2(ψ[ix, iy, iz]); end
            rho[iz] = a * g.Δx * g.Δy
        end
        return [g.Z[iz+1] for iz in 1:Nz], rho
    elseif dir == :x
        rho = zeros(Nx)
        @inbounds for ix in 1:Nx
            a = 0.0
            for iy in 1:Ny, iz in 1:Nz; a += abs2(ψ[ix, iy, iz]); end
            rho[ix] = a * g.Δy * g.Δz
        end
        return [g.X[ix+1] for ix in 1:Nx], rho
    else
        rho = zeros(Ny)
        @inbounds for iy in 1:Ny
            a = 0.0
            for ix in 1:Nx, iz in 1:Nz; a += abs2(ψ[ix, iy, iz]); end
            rho[iy] = a * g.Δx * g.Δz
        end
        return [g.Y[iy+1] for iy in 1:Ny], rho
    end
end

function trapz(x::AbstractVector, y::AbstractVector)
    n = length(x); n < 2 && return 0.0
    s = 0.0
    @inbounds for i in 1:n-1
        s += 0.5 * (x[i+1] - x[i]) * (y[i+1] + y[i])
    end
    return s
end

function split_density_at_omega(s, rho, s0, tsign)
    idx = tsign > 0 ? findall(x -> x >= s0, s) : findall(x -> x <= s0, s)
    isempty(idx) && return (0.0, trapz(s, rho))
    Pc = trapz(s[idx], rho[idx])
    return (Pc, trapz(s, rho) - Pc)
end

# ════════════════════════════════════════════════════════════════
#  ÁNGULO DE DISPERSIÓN (Cabrera-Trujillo et al.)
#
#  θ = cos⁻¹(P_i · P_f / |P_i| |P_f|)
#  Signo: componente de P_f en DIR_B positiva → repulsión (+)
#                                   negativa → atracción (-)
# ════════════════════════════════════════════════════════════════

function scattering_angle(Pi::Vector{Float64}, Pf::Vector{Float64})
    mag_i = sqrt(Pi[1]^2 + Pi[2]^2 + Pi[3]^2)
    mag_f = sqrt(Pf[1]^2 + Pf[2]^2 + Pf[3]^2)
    if mag_i < 1e-14 || mag_f < 1e-14
        return (θ_rad=NaN, θ_deg=NaN, Pi_mag=mag_i, Pf_mag=mag_f)
    end
    dot_pp = Pi[1]*Pf[1] + Pi[2]*Pf[2] + Pi[3]*Pf[3]
    cos_θ = clamp(dot_pp / (mag_i * mag_f), -1.0, 1.0)
    θ = acos(cos_θ)
    # Signo por componente en eje del parámetro de impacto
    ab = ax_idx(DIR_B)
    if Pf[ab] < 0.0
        θ = -θ
    end
    return (θ_rad=θ, θ_deg=rad2deg(θ), Pi_mag=mag_i, Pf_mag=mag_f)
end

# ════════════════════════════════════════════════════════════════
#  SELECCIÓN DE PROCESO
# ════════════════════════════════════════════════════════════════

function pick_process(ZP_val::Float64, mode::Symbol)
    if mode == :auto
        ZP_val > 0 && return (:capture, true, false)
        ZP_val < 0 && return (:ionize, false, true)
        return (:both, true, true)
    end
    mode == :capture && return (:capture, true, false)
    mode == :ionize  && return (:ionize, false, true)
    mode == :both    && return (:both, true, true)
    error("PROCESS_MODE inválido: $mode")
end

# ════════════════════════════════════════════════════════════════
#  SALIDA DE DATOS
# ════════════════════════════════════════════════════════════════

function dump_density_yz(fn, g, ψ; t::Float64=NaN,
                         R::Union{Matrix{Float64},Nothing}=nothing,
                         Rcm::Union{Tuple{Float64,Float64,Float64},Nothing}=nothing)
    open(fn, "w") do io
        println(io, "# dens_yz: rho(y,z) = integral dx |psi|^2")
        @printf(io, "# t = %.6f\n", isnan(t) ? -1.0 : t)
        if R !== nothing
            @printf(io, "# R1 = %.6f %.6f %.6f\n", R[1,1], R[1,2], R[1,3])
            @printf(io, "# R2 = %.6f %.6f %.6f\n", R[2,1], R[2,2], R[2,3])
        end
        if Rcm !== nothing
            @printf(io, "# Rcm = %.6f %.6f %.6f\n", Rcm[1], Rcm[2], Rcm[3])
        end
        println(io, "# columns: y  z  rho")
        @inbounds for iy in 1:g.Ny, iz in 1:g.Nz
            s = 0.0
            for ix in 1:g.Nx; s += abs2(ψ[ix, iy, iz]); end
            @printf(io, "%.8e %.8e %.8e\n", g.Y[iy+1], g.Z[iz+1], s * g.Δx)
        end
    end
end

function dump_density_z(fn, g, ψ)
    open(fn, "w") do io
        println(io, "# dens_z: rho(z) = integral dx dy |psi|^2")
        println(io, "# columns: z  rho")
        @inbounds for iz in 1:g.Nz
            s = 0.0
            for ix in 1:g.Nx, iy in 1:g.Ny; s += abs2(ψ[ix, iy, iz]); end
            @printf(io, "%.8e %.8e\n", g.Z[iz+1], s * g.Δx * g.Δy)
        end
    end
end

function dump_momentum_kz(fn, ψ, g)
    Nx, Ny, Nz = size(ψ)
    ρkz = zeros(Float64, Nz); line = Vector{ComplexF64}(undef, Nz)
    @inbounds for ix in 1:Nx, iy in 1:Ny
        for iz in 1:Nz; line[iz] = ψ[ix, iy, iz]; end
        sp = fft(line)
        for j in 1:Nz; ρkz[j] += abs2(sp[j]); end
    end
    Δkz = 2π / (Nz * g.Δz); h = Nz ÷ 2
    ρs = vcat(ρkz[h+1:end], ρkz[1:h])
    kz = [(j - 1 - h) * Δkz for j in 1:Nz]
    nk = sum(ρs) * Δkz
    if nk > 0; ρs ./= nk; end
    open(fn, "w") do io
        println(io, "# dens_kz: rho(k_z)")
        println(io, "# columns: kz  rho_kz")
        for j in 1:Nz
            @printf(io, "%.12e %.12e\n", kz[j], ρs[j])
        end
    end
end

function write_energy_line(io, t, E)
    @printf(io, "%12.6f  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %12.8f  %16.8e  %16.8e  %16.8e  %16.8e\n",
            t, E.K_T, E.K_P, E.V_nn, E.T_e, E.V_eN,
            E.E_elec, E.E_total, E.norm,
            E.T_e_norm, E.V_eN_norm, E.E_elec_norm, E.E_total_norm)
end

function write_traj_line(io, t, R, P, rcm)
    @printf(io, "%12.6f  % .6f % .6f % .6f  % .6f % .6f % .6f  % .6f % .6f % .6f  % .6f % .6f % .6f  % .8f % .8f % .8f\n",
            t, R[1,1], R[1,2], R[1,3], R[2,1], R[2,2], R[2,3],
            P[1,1], P[1,2], P[1,3], P[2,1], P[2,2], P[2,3],
            rcm[1], rcm[2], rcm[3])
end

function write_info_txt(path, title, entries)
    open(path, "w") do io
        println(io, "$CODE_NAME v$CODE_VERSION  |  $title")
        println(io, "Generado: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "─"^60)
        for (k, v) in entries
            println(io, k, ": ", v)
        end
    end
end

# ════════════════════════════════════════════════════════════════
#  CONDICIONES INICIALES
# ════════════════════════════════════════════════════════════════

function initial_conditions(b::Float64, E_keV::Float64, sys::PhysSystem)
    p = momentum_from_keV(E_keV, sys.M_proj)
    R = zeros(Float64, 2, 3)
    P = zeros(Float64, 2, 3)
    R[1,:] .= R_TARGET_INIT
    ac = ax_idx(DIR_COLLISION)
    ab = ax_idx(DIR_B)
    R[2, ac] = S_TRAJ[1]
    R[2, ab] = R_TARGET_INIT[ab] + b
    P[2, ac] = p
    return R, P, p
end

# ════════════════════════════════════════════════════════════════
#  DRIVER: UNA SIMULACIÓN
# ════════════════════════════════════════════════════════════════

function run_one!(sim_dir::String, b::Float64, E_keV::Float64,
                  g::Grid3D, sys::PhysSystem, kin::KineticOps,
                  Mab::Array{Float64,3}, dt::Float64,
                  DO_CAP::Bool, DO_ION::Bool)

    tw0 = time_ns()
    masses = [sys.M_target, sys.M_proj]
    R, P, p0 = initial_conditions(b, E_keV, sys)
    P2_init = copy(P[2,:])   # momento inicial del proyectil (para θ)

    ac = ax_idx(DIR_COLLISION)
    tsign = sign(S_TRAJ[2] - S_TRAJ[1])
    v0 = P[2, ac] / sys.M_proj
    Tg = abs(S_TRAJ[2] - S_TRAJ[1]) / max(abs(v0), 1e-12)
    nmax = Int(ceil(Tg / dt))

    # ── Campos ──
    ψ = Array{ComplexF64,3}(undef, g.Nx, g.Ny, g.Nz)
    V = Array{Float64,3}(undef, g.Nx, g.Ny, g.Nz)
    init_psi_1s!(ψ, g, R_TARGET_INIT, sys.ZT)
    normalize!(ψ, g.dV)

    # ── Energías iniciales ──
    Ei = compute_energies(ψ, V, g, R, P, sys)
    KPi = Ei.K_P

    # ── CN LU ──
    cdt0 = 1im * (dt / 2)
    LUx = build_cn_lu(kin.Hx_d, kin.Hx_o, cdt0)
    LUy = build_cn_lu(kin.Hy_d, kin.Hy_o, cdt0)
    LUz = build_cn_lu(kin.Hz_d, kin.Hz_o, cdt0)

    # ── Archivos de log ──
    io_tr = nothing
    if WRITE_TRAJ_LOG
        io_tr = open(joinpath(sim_dir, "traj_log.dat"), "w")
        println(io_tr, "# $CODE_NAME v$CODE_VERSION | E=$(E_keV)keV b=$b")
        println(io_tr, "# t  R1x R1y R1z  R2x R2y R2z  P1x P1y P1z  P2x P2y P2z  CMx CMy CMz")
        write_traj_line(io_tr, 0.0, R, P, center_of_mass(R, sys))
    end

    io_el = nothing
    do_el = WRITE_ENERGY_LOG && ENERGY_LOG_EVERY > 0
    if do_el
        io_el = open(joinpath(sim_dir, "energy_log.dat"), "w")
        println(io_el, "# t  K_T  K_P  V_nn  T_e  V_eN  E_elec  E_total  norm  T_e_norm  V_eN_norm  E_elec_norm  E_total_norm")
        write_energy_line(io_el, 0.0, Ei)
    end

    snap_dir = joinpath(sim_dir, "datos_dens_yz")
    if WRITE_SNAPSHOTS; mkpath(snap_dir); end

    # ── Progreso en terminal ──
    t_wall_loop = time_ns()
    @printf("  TMAX = %.1f a.u.  |  nsteps = %d\n", Tg, nmax)
    println("  ──────────────────────────────────────────────────────────")
    @printf("  %-12s %-10s %-10s %-10s %-10s %-6s  %s\n",
            "Paso", "t(a.u.)", "R₁₂", "v_proj", "norma", "%", "est")
    println("  ──────────────────────────────────────────────────────────")

    # ── Bucle temporal ──
    t = 0.0; nsteps = 0; nfr = 0; r12_min = Inf

    for k in 1:nmax
        nsteps = k
        do_one_step!(R, P, ψ, V, g, sys, masses, Mab, kin,
                     LUx, LUy, LUz, dt, k)
        t += dt

        dx12 = R[2,1] - R[1,1]; dy12 = R[2,2] - R[1,2]; dz12 = R[2,3] - R[1,3]
        r12_now = sqrt(dx12^2 + dy12^2 + dz12^2)
        r12_min = min(r12_min, r12_now)

        # ── Traj log ──
        if io_tr !== nothing
            write_traj_line(io_tr, t, R, P, center_of_mass(R, sys))
        end

        # ── Energy log ──
        if do_el && mod(k, ENERGY_LOG_EVERY) == 0
            write_energy_line(io_el, t, compute_energies(ψ, V, g, R, P, sys))
        end

        # ── Snapshots ──
        if WRITE_SNAPSHOTS && mod(k, SNAP_EVERY) == 0
            nfr += 1
            rcm = center_of_mass(R, sys)
            dump_density_yz(joinpath(snap_dir, @sprintf("dens_yz_%05d.dat", nfr)),
                            g, ψ; t=t, R=R, Rcm=rcm)
        end

        # ── Progreso en terminal ──
        if mod(k, PROGRESS_EVERY) == 0 || k == nmax
            pct = round(Int, 100.0 * k / nmax)
            nrm_now = sum(abs2, ψ) * g.dV
            vp_now = sqrt(P[2,1]^2 + P[2,2]^2 + P[2,3]^2) / sys.M_proj
            elapsed_loop = (time_ns() - t_wall_loop) / 1e9
            if k < nmax
                rate = elapsed_loop / k
                est_sec = rate * (nmax - k)
                est_str = fmt_time(est_sec)
            else
                est_str = "—"
            end
            @printf("  %5d/%-5d  %8.1f  %8.3f  %10.6f  %8.5f  %3d%%  ▸ %s\n",
                    k, nmax, t, r12_now, vp_now, nrm_now, pct, est_str)
        end
    end

    println("  ──────────────────────────────────────────────────────────")
    if io_tr !== nothing; close(io_tr); end

    # ── Energías finales ──
    Ef = compute_energies(ψ, V, g, R, P, sys)
    if do_el
        write_energy_line(io_el, t, Ef)
        close(io_el)
    end

    ΔEp = Ef.K_P - KPi
    KTr = Ef.K_T
    ΔEt = Ef.E_total - Ei.E_total
    nf  = Ef.norm

    # ── Ángulo de dispersión (Cabrera-Trujillo) ──
    P2_final = [P[2,1], P[2,2], P[2,3]]
    scat = scattering_angle(P2_init, P2_final)

    # ── Pcap / Pion ──
    s, rho = axis_density(ψ, g, DIR_COLLISION)
    sT = R[1, ac]; sP_f = R[2, ac]
    s0 = OMEGA

    Pcap = NaN; Ptar = NaN; Pion = NaN
    if DO_CAP
        Pcap, Ptar = split_density_at_omega(s, rho, s0, tsign)
    end
    if DO_ION
        Pion = max(1.0 - nf, 0.0)
    end

    # ── Densidades finales ──
    dump_density_z(joinpath(sim_dir, "dens_final_z.dat"), g, ψ)
    dump_density_yz(joinpath(sim_dir, "dens_final_yz.dat"), g, ψ;
                    t=t, R=R, Rcm=center_of_mass(R, sys))
    dump_momentum_kz(joinpath(sim_dir, "dens_final_kz.dat"), ψ, g)

    wsec = (time_ns() - tw0) / 1e9

    # ── run_info.txt ──
    entries = Pair{String,Any}[
        "dir_collision"  => String(DIR_COLLISION),
        "dir_b"          => String(DIR_B),
        "E_keV"          => E_keV,
        "b"              => b,
        "S_TRAJ"         => string(S_TRAJ),
        "─── Malla y absorbente ───" => "",
        "domain"         => @sprintf("[%.1f,%.1f]×[%.1f,%.1f]×[%.1f,%.1f]",
                                     XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX),
        "mesh"           => @sprintf("%d×%d×%d", NX, NY, NZ),
        "Δ"              => @sprintf("%.6f, %.6f, %.6f", g.Δx, g.Δy, g.Δz),
        "mask_type"      => "cos^(1/8)",
        "mask_Lx_abs"    => LX_ABS,
        "mask_Ly_abs"    => LY_ABS,
        "mask_Lz_abs"    => LZ_ABS,
        "mask_summary"   => @sprintf("cos^(1/8), Lx=%.3f, Ly=%.3f, Lz=%.3f a.u.",
                                 LX_ABS, LY_ABS, LZ_ABS),
        "p0"             => p0,
        "v0"             => v0,
        "dt"             => dt,
        "nsteps"         => nsteps,
        "t_final"        => t,
        "sP_final"       => sP_f,
        "sT_final"       => sT,
        "r12_min"        => r12_min,
        "─── Dispersión (Cabrera-Trujillo) ───" => "",
        "Pi"             => string(P2_init),
        "Pf"             => string(P2_final),
        "Pi_mag"         => scat.Pi_mag,
        "Pf_mag"         => scat.Pf_mag,
        "theta_rad"      => scat.θ_rad,
        "theta_deg"      => scat.θ_deg,
        "─── Observables ───" => "",
        "Omega"          => s0,
        "Pcap"           => Pcap,
        "Ptar"           => Ptar,
        "Pion"           => Pion,
        "norm_final"     => nf,
        "nframes"        => nfr,
        "─── E_init ───" => "",
        "K_T_i"          => Ei.K_T,
        "K_P_i"          => Ei.K_P,
        "V_nn_i"         => Ei.V_nn,
        "E_total_i"      => Ei.E_total,
        "─── E_final ───" => "",
        "K_T_f"          => Ef.K_T,
        "K_P_f"          => Ef.K_P,
        "V_nn_f"         => Ef.V_nn,
        "E_total_f"      => Ef.E_total,
        "─── Balance ───" => "",
        "ΔE_proj"        => ΔEp,
        "K_T_recoil"     => KTr,
        "ΔE_total"       => ΔEt,
        "wall_sec"       => wsec,
    ]
    write_info_txt(joinpath(sim_dir, "run_info.txt"), "Run info", entries)

    # ── Resumen en terminal ──
    θ_str = isnan(scat.θ_deg) ? "NaN" : @sprintf("%.2f°", scat.θ_deg)
    @printf("  ✓ Completada en %s  |  norm = %.6f  |  θ = %s\n",
            fmt_time(wsec), nf, θ_str)

    return (Pcap=Pcap, Ptar=Ptar, Pion=Pion, norm_final=nf,
            s0=s0, t_final=t, nsteps=nsteps, sP_final=sP_f,
            r12_min=r12_min,
            θ_rad=scat.θ_rad, θ_deg=scat.θ_deg,
            Pi_mag=scat.Pi_mag, Pf_mag=scat.Pf_mag,
            nframes=nfr,
            ΔE_proj=ΔEp, K_T_recoil=KTr, ΔE_total=ΔEt,
            wall_sec=wsec)
end

# ════════════════════════════════════════════════════════════════
#  SUITE PRINCIPAL
# ════════════════════════════════════════════════════════════════

function main_suite()
    tw0 = time_ns()
    sys = PhysSystem(ZT, ZP, M_TARGET, M_PROJ, M_EFF)
    g = build_grid(XMIN, XMAX, NX, YMIN, YMAX, NY, ZMIN, ZMAX, NZ)
    kin = build_kinetic_ops(sys, g)
    Mab = build_absorbing_mask(g, LX_ABS, LY_ABS, LZ_ABS)
    (tag, DO_CAP, DO_ION) = pick_process(sys.ZP, PROCESS_MODE)
    sn = system_label(sys)
    nsim = length(E_KEV_LIST) * length(B_LIST)

    println("═"^66)
    println("  $CODE_NAME v$CODE_VERSION — Electron-Nuclear Dynamics")
    println("  Sistema: $sn  |  Modo: $tag")
    if !isempty(USER_NOTE); println("  Nota: $USER_NOTE"); end
    println("═"^66)
    @printf("  Malla: %d×%d×%d = %d pts  (Δ=%.4f, %.4f, %.4f)\n",
            g.Nx, g.Ny, g.Nz, g.Nx * g.Ny * g.Nz, g.Δx, g.Δy, g.Δz)
    @printf("  Absorbente: cos^(1/8) | Lx=%.1f, Ly=%.1f, Lz=%.1f a.u.\n",
        LX_ABS, LY_ABS, LZ_ABS)
    @printf("  Colisión: dir=%s, b en %s | S=%.1f → %.1f\n",
            DIR_COLLISION, DIR_B, S_TRAJ[1], S_TRAJ[2])
    @printf("  Energías: %d  |  b: %d  →  %d sims\n",
            length(E_KEV_LIST), length(B_LIST), nsim)
    println("  ADI alternante (z→y→x / x→y→z)")
    println("═"^66)

    sim_count = 0

    for E_keV in E_KEV_LIST
        pt = momentum_from_keV(E_keV, sys.M_proj)
        vt = pt / sys.M_proj
        dt = compute_dt(g.Δx, g.Δy, g.Δz, vt)

        println()
        @printf("▶ E = %s  (v₀=%.6f, Δt=%.6f)\n",
                energy_folder_name(E_keV), vt, dt)
        println("═"^66)

        for b in B_LIST
            sim_count += 1

            # ── Construir ruta ──
            sim_dir, e_dir = build_sim_path(sys, E_keV, b)

            @printf("\n  ● b = %.1f a.u.  [%d/%d]\n", b, sim_count, nsim)
            println("  Carpeta: $sim_dir")

            res = run_one!(sim_dir, b, E_keV, g, sys, kin, Mab, dt,
                           DO_CAP, DO_ION)

            # ── Append a observables.dat (por energía) ──
            obs_path = joinpath(e_dir, "observables.dat")
            if !isfile(obs_path)
                open(obs_path, "w") do io
                    println(io, "# b  Pcap  Ptar  Pion  norm  bPcap  bPion  Omega  theta_deg  Pi_mag  Pf_mag  t_final  nsteps  nframes  sP_final  r12_min  dE_proj  KT_rec  dE_total  wall_sec")
                end
            end

            bPc = isnan(res.Pcap) ? NaN : b * res.Pcap
            bPi = isnan(res.Pion) ? NaN : b * res.Pion
            open(obs_path, "a") do io
                @printf(io, "%10.6f  %12.6e  %12.6e  %12.6e  %12.8f  %12.6e  %12.6e  %10.6f  %10.4f  %16.8e  %16.8e  %10.4f  %6d  %6d  %10.4f  %10.4f  %16.8e  %16.8e  %16.8e  %10.4f\n",
                        b, res.Pcap, res.Ptar, res.Pion, res.norm_final,
                        isnan(bPc) ? NaN : bPc, isnan(bPi) ? NaN : bPi,
                        res.s0, res.θ_deg, res.Pi_mag, res.Pf_mag,
                        res.t_final, res.nsteps, res.nframes,
                        res.sP_final, res.r12_min,
                        res.ΔE_proj, res.K_T_recoil, res.ΔE_total,
                        res.wall_sec)
            end
        end
    end

    wt = (time_ns() - tw0) / 1e9

    # ── runset_info.txt en Colisiones ──
    sn = system_label(sys)
    anim_dir = joinpath(BASE_PATH, sn, "Colisiones")
    entries = Pair{String,Any}[
        "code"           => "$CODE_NAME v$CODE_VERSION",
        "system"         => sn,
        "process"        => string(tag),
        "note"           => USER_NOTE,
        "ZT"             => ZT,
        "ZP"             => ZP,
        "M_target"       => M_TARGET,
        "M_proj"         => M_PROJ,

        "DIR_COLLISION"  => String(DIR_COLLISION),
        "DIR_B"          => String(DIR_B),
        "S_TRAJ"         => string(S_TRAJ),

        "domain"         => @sprintf("[%.1f,%.1f]×[%.1f,%.1f]×[%.1f,%.1f]",
                                     XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX),
        "mesh"           => @sprintf("%d×%d×%d", NX, NY, NZ),
        "Δ"              => @sprintf("%.6f, %.6f, %.6f", g.Δx, g.Δy, g.Δz),

        "mask_type"      => "cos^(1/8)",
        "mask_Lx_abs"    => LX_ABS,
        "mask_Ly_abs"    => LY_ABS,
        "mask_Lz_abs"    => LZ_ABS,
        "mask_summary"   => @sprintf("cos^(1/8), Lx=%.3f, Ly=%.3f, Lz=%.3f a.u.",
                                     LX_ABS, LY_ABS, LZ_ABS),

        "OMEGA"          => OMEGA,
        "total_runs"     => sim_count,
        "wall_time"      => @sprintf("%.2f s (%s)", wt, fmt_time(wt)),
    ]
    write_info_txt(joinpath(anim_dir, "runset_info.txt"), "Global runset info", entries)

    println("\n", "═"^66)
    println("  RESUMEN FINAL — $CODE_NAME v$CODE_VERSION")
    @printf("  Simulaciones:  %d\n", sim_count)
    @printf("  Tiempo total:  %s\n", fmt_time(wt))
    println("═"^66)
end

# ════════════════════════════════════════════════════════════════
const THIS_FILE = abspath(@__FILE__)
if (abspath(PROGRAM_FILE) == THIS_FILE) || isinteractive()
    Base.invokelatest(main_suite)
end
