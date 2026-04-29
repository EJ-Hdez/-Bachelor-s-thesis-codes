# ════════════════════════════════════════════════════════════════
# gauss3d_cn_yz.jl  v1.0.0
# Paquete gaussiano 3D libre — Crank-Nicolson ADI
#
# Resuelve la TDSE libre en u.a. (ħ = m = 1, V = 0):
#   i ∂t Ψ = -1/2 ∇² Ψ
# en un dominio 3D con frontera Dirichlet (Ψ = 0 en los bordes).
#
# Esquema (un paso completo Δt):
#   ADI: (I - ν_d δ²_d) Ψ^{n+1} = (I + ν_d δ²_d) Ψ^n  por cada dirección
#   con  ν_d = i Δt / (4 Δd²)
#   - Pasos impares: x → y → z
#   - Pasos pares:   z → y → x
#   La alternancia cancela el sesgo direccional ADI a O(Δt²).
#
# Notas físico-numéricas:
#   • CN unitario: la norma se conserva a doble precisión.
#   • C_d = ħ Δt /(4 m Δd²) son los módulos |ν_d|. Para precisión
#     conviene C_d ≲ 1; valores muy grandes degradan la fase, no
#     desestabilizan el método.
#   • Dirichlet 0: refleja el paquete si llega a la frontera.
#
# Uso:
#   julia gauss3d_cn_yz_v2.jl
# ════════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Dates
using DelimitedFiles
using CairoMakie

const CODE_NAME    = "gauss3d_cn_yz_v2.jl"
const CODE_VERSION = "1.0.0"

# ════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DEL USUARIO
# ════════════════════════════════════════════════════════════════

const OUT_BASE = "/media/usuario/your/path/Gaussiana 3D"

# ── Constantes físicas (u.a.) ─────────────────────────────────
const HBAR = 1.0
const M    = 1.0

# ── Dominio ───────────────────────────────────────────────────
const LX, LY, LZ = 15.0, 15.0, 35.0
# x ∈ [-LX/2, LX/2], y ∈ [-LY/2, LY/2], z ∈ [0, LZ]

# ── Malla ─────────────────────────────────────────────────────
# Recomendada para tesis (corrida ~10 min en laptop):
const NX, NY, NZ = 121, 121, 241
# Alternativa rápida (~1-2 min):
# const NX, NY, NZ = 101, 101, 201

# ── Paso temporal ─────────────────────────────────────────────
const DT_USER = 0.01
const T_FINAL = 7.0

# ── Condición inicial: gaussiana centrada con momento +k0 ẑ ──
const X0 = 0.0
const Y0 = 0.0
const Z0 = 0.25 * LZ          # 8.75 con LZ=35
const SIGMA_X = 2.5
const SIGMA_Y = 2.5
const SIGMA_Z = 2.5
const K0 = 2.0                # momento medio en +z (vg ≈ k0)

# ── Tiempos de captura ────────────────────────────────────────
const SNAP_TIMES = [0.0, 4.30, 7.00]

# ── Animación (opcional) ──────────────────────────────────────
const GEN_ANIMATION  = true
const ANIM_FORMAT    = :mp4
const ANIM_EVERY     = 5
const ANIM_FRAMERATE = 30

# ── Estilo visual ─────────────────────────────────────────────
const COLORMAP   = :hot
const SAVE_PDF   = true
const SAVE_PNG   = true
const DPI        = 300
const FONT_SIZE  = 18
const TITLE_SIZE = 22

# ── Progreso ──────────────────────────────────────────────────
const PROGRESS_EVERY = 25

# ════════════════════════════════════════════════════════════════
#  TEMA VISUAL
# ════════════════════════════════════════════════════════════════

function set_thesis_theme!()
    set_theme!(Theme(
        fontsize = FONT_SIZE,
        Axis = (
            xgridvisible = false, ygridvisible = false,
            xlabelsize = FONT_SIZE + 2, ylabelsize = FONT_SIZE + 2,
            xticklabelsize = FONT_SIZE - 2, yticklabelsize = FONT_SIZE - 2,
            titlesize = TITLE_SIZE, spinewidth = 1.4,
            xtickwidth = 1.2, ytickwidth = 1.2,
            topspinevisible = true, rightspinevisible = true,
        ),
        Legend = (framevisible = false, labelsize = FONT_SIZE),
    ))
end

# ════════════════════════════════════════════════════════════════
#  UTILIDADES
# ════════════════════════════════════════════════════════════════

ensure_dir(p) = (isdir(p) || mkpath(p); p)

# 0.000 → "0p000"
fmt_t_tag(t) = replace(@sprintf("%.3f", t), "." => "p")

function savefig(fig, dir, name)
    ensure_dir(dir)
    SAVE_PDF && save(joinpath(dir, name * ".pdf"), fig)
    SAVE_PNG && save(joinpath(dir, name * ".png"), fig; px_per_unit = DPI / 96)
end

function fmt_time(s::Float64)
    s < 60   && return @sprintf("%.1fs", s)
    s < 3600 && return @sprintf("%dm%02ds", div(Int(floor(s)),60), mod(Int(floor(s)),60))
    return @sprintf("%dh%02dm", div(Int(floor(s)),3600), div(mod(Int(floor(s)),3600),60))
end

function nearest_index(arr, x)
    _, i = findmin(abs.(arr .- x))
    return i
end

# ════════════════════════════════════════════════════════════════
#  MALLA
# ════════════════════════════════════════════════════════════════

function build_grid()
    x = collect(range(-LX/2, LX/2, length = NX))
    y = collect(range(-LY/2, LY/2, length = NY))
    z = collect(range(0.0,  LZ,    length = NZ))
    dx = x[2] - x[1]; dy = y[2] - y[1]; dz = z[2] - z[1]
    return x, y, z, dx, dy, dz
end

# ════════════════════════════════════════════════════════════════
#  CONDICIÓN INICIAL Y NORMALIZACIÓN
# ════════════════════════════════════════════════════════════════

function gaussian_packet(x, y, z)
    psi = Array{ComplexF64,3}(undef, NX, NY, NZ)
    inv2sx2 = 1.0 / (2.0 * SIGMA_X^2)
    inv2sy2 = 1.0 / (2.0 * SIGMA_Y^2)
    inv2sz2 = 1.0 / (2.0 * SIGMA_Z^2)
    @inbounds for k in 1:NZ
        dz2 = (z[k] - Z0)^2
        phase_k = K0 * (z[k] - Z0)
        for j in 1:NY
            dy2 = (y[j] - Y0)^2
            for i in 1:NX
                dx2 = (x[i] - X0)^2
                env = exp(-(dx2*inv2sx2 + dy2*inv2sy2 + dz2*inv2sz2))
                psi[i,j,k] = env * cis(phase_k)
            end
        end
    end
    # Frontera Dirichlet: Ψ = 0 en los bordes de la malla
    psi[1,  :, :] .= 0; psi[NX, :, :] .= 0
    psi[:,  1, :] .= 0; psi[:, NY, :] .= 0
    psi[:,  :, 1] .= 0; psi[:, :, NZ] .= 0
    return psi
end

function norm_squared(psi, dV)
    s = 0.0
    @inbounds for v in psi; s += abs2(v); end
    return s * dV
end

function normalize!(psi, dV)
    N = norm_squared(psi, dV)
    N > 0 && (psi ./= sqrt(N))
    return psi
end

# ════════════════════════════════════════════════════════════════
#  OBSERVABLES
# ════════════════════════════════════════════════════════════════

"""ρ_yz(y, z) = ∫ |Ψ|² dx  →  Matriz [NY, NZ]."""
function rho_yz(psi, dx)
    rho = zeros(Float64, NY, NZ)
    @inbounds for k in 1:NZ, j in 1:NY
        s = 0.0
        for i in 1:NX
            s += abs2(psi[i,j,k])
        end
        rho[j,k] = s * dx
    end
    return rho
end

"""<x>, <y>, <z> con suma de Riemann."""
function expectation_position(psi, x, y, z, dV)
    sx = 0.0; sy = 0.0; sz = 0.0
    @inbounds for k in 1:NZ
        zk = z[k]
        for j in 1:NY
            yj = y[j]
            for i in 1:NX
                p = abs2(psi[i,j,k])
                sx += x[i] * p
                sy += yj   * p
                sz += zk   * p
            end
        end
    end
    return sx*dV, sy*dV, sz*dV
end

"""Laplaciano discreto en `out`. Usa frontera Dirichlet implícita
   (Ψ = 0 fuera del dominio): los puntos de borde de la malla
   contribuyen con sus vecinos del interior y un cero virtual fuera."""
function laplacian!(out::Array{ComplexF64,3}, psi::Array{ComplexF64,3},
                    dx::Float64, dy::Float64, dz::Float64)
    invdx2 = 1.0 / dx^2
    invdy2 = 1.0 / dy^2
    invdz2 = 1.0 / dz^2
    @inbounds for k in 1:NZ
        for j in 1:NY
            for i in 1:NX
                p = psi[i,j,k]
                # vecinos x
                lx_p = i < NX ? psi[i+1,j,k] : zero(ComplexF64)
                lx_m = i > 1  ? psi[i-1,j,k] : zero(ComplexF64)
                # vecinos y
                ly_p = j < NY ? psi[i,j+1,k] : zero(ComplexF64)
                ly_m = j > 1  ? psi[i,j-1,k] : zero(ComplexF64)
                # vecinos z
                lz_p = k < NZ ? psi[i,j,k+1] : zero(ComplexF64)
                lz_m = k > 1  ? psi[i,j,k-1] : zero(ComplexF64)
                out[i,j,k] = (lx_p - 2p + lx_m) * invdx2 +
                             (ly_p - 2p + ly_m) * invdy2 +
                             (lz_p - 2p + lz_m) * invdz2
            end
        end
    end
    return out
end

"""Energía cinética <H> = ∫ Ψ* (-1/2 ∇²Ψ) dV.
   Devuelve la parte real (la imaginaria es ≈0 por hermiticidad)."""
function expectation_energy(psi, lap_buf, dx, dy, dz, dV)
    laplacian!(lap_buf, psi, dx, dy, dz)
    s = 0.0 + 0.0im
    @inbounds for I in eachindex(psi)
        s += conj(psi[I]) * lap_buf[I]
    end
    s *= dV
    return -0.5 * (HBAR^2 / M) * real(s)
end

# ════════════════════════════════════════════════════════════════
#  CRANK-NICOLSON ADI
# ════════════════════════════════════════════════════════════════

"""Construye A_plus (LHS) tridiagonal compleja con frontera Dirichlet.
   - Filas interiores: diag = 1 + 2ν, off = -ν
   - Filas de borde:   diag = 1, off = 0  (mantiene Ψ_borde = 0)"""
function build_LHS(N::Int, nu::ComplexF64)
    diag = fill(1 + 2nu, N)
    sup  = fill(-nu, N - 1)
    sub  = fill(-nu, N - 1)
    diag[1] = 1 + 0im; sup[1]   = 0 + 0im
    diag[N] = 1 + 0im; sub[N-1] = 0 + 0im
    return Tridiagonal(sub, diag, sup)
end

"""Aplica A_minus = I + ν δ² al vector línea ψ → out (Dirichlet 0)."""
@inline function apply_A_minus!(out::Vector{ComplexF64},
                                psi_line::Vector{ComplexF64},
                                nu::ComplexF64)
    N = length(psi_line)
    @inbounds begin
        out[1] = 0 + 0im
        for i in 2:N-1
            out[i] = (1 - 2nu) * psi_line[i] +
                     nu * (psi_line[i-1] + psi_line[i+1])
        end
        out[N] = 0 + 0im
    end
end

# ── Sweeps direccionales ───────────────────────────────────────

function sweep_x!(psi, LU_x, nu_x, line, rhs)
    @inbounds for k in 1:NZ, j in 1:NY
        for i in 1:NX; line[i] = psi[i,j,k]; end
        apply_A_minus!(rhs, line, nu_x)
        ldiv!(LU_x, rhs)
        for i in 1:NX; psi[i,j,k] = rhs[i]; end
    end
end

function sweep_y!(psi, LU_y, nu_y, line, rhs)
    @inbounds for k in 1:NZ, i in 1:NX
        for j in 1:NY; line[j] = psi[i,j,k]; end
        apply_A_minus!(rhs, line, nu_y)
        ldiv!(LU_y, rhs)
        for j in 1:NY; psi[i,j,k] = rhs[j]; end
    end
end

function sweep_z!(psi, LU_z, nu_z, line, rhs)
    @inbounds for j in 1:NY, i in 1:NX
        for k in 1:NZ; line[k] = psi[i,j,k]; end
        apply_A_minus!(rhs, line, nu_z)
        ldiv!(LU_z, rhs)
        for k in 1:NZ; psi[i,j,k] = rhs[k]; end
    end
end

"""Un paso Δt completo, con orden ADI alternante."""
function step_CN_ADI!(psi, LU_x, LU_y, LU_z, nu_x, nu_y, nu_z, n_step,
                      line_x, rhs_x, line_y, rhs_y, line_z, rhs_z)
    if isodd(n_step)
        sweep_x!(psi, LU_x, nu_x, line_x, rhs_x)
        sweep_y!(psi, LU_y, nu_y, line_y, rhs_y)
        sweep_z!(psi, LU_z, nu_z, line_z, rhs_z)
    else
        sweep_z!(psi, LU_z, nu_z, line_z, rhs_z)
        sweep_y!(psi, LU_y, nu_y, line_y, rhs_y)
        sweep_x!(psi, LU_x, nu_x, line_x, rhs_x)
    end
end

# ════════════════════════════════════════════════════════════════
#  FIGURAS
# ════════════════════════════════════════════════════════════════

"""Captura individual ρ_yz(y, z, t)."""
function save_snapshot_yz(fig_dir, t, y, z, rho, vmax)
    fig = Figure(size = (920, 420))
    ax = Axis(fig[1,1];
    xlabel = "z (u.a.)",
    ylabel = "y (u.a.)",
    title  = @sprintf("ρ_yz(y, z, t)    t = %.2f", t),
    aspect = DataAspect(),
)
    hm = heatmap!(ax, z, y, permutedims(rho);
                  colormap = COLORMAP, colorrange = (0.0, vmax))
    Colorbar(fig[1, 2], hm; label = "ρ_yz")
    savefig(fig, fig_dir, "rho_yz_t_$(fmt_t_tag(t))")
end


function save_panel_yz(fig_dir, ts_list, y, z, rho_list, vmax)
    # Asegura orden temporal ascendente
    p = sortperm(ts_list)
    ts_sorted  = ts_list[p]
    rho_sorted = rho_list[p]

    n = length(ts_sorted)

    # Ancho moderado y altura proporcional al número de paneles
    fig = Figure(size = (980, 320 * n))

    hm_ref = nothing

    for k in 1:n
        ax = Axis(fig[k, 1];
            xlabel = (k == n ? "z (u.a.)" : ""),
            ylabel = "y (u.a.)",
            title  = @sprintf("ρ_yz(y, z, t)    t = %.2f", ts_sorted[k]),
            aspect = DataAspect(),
        )

        hm = heatmap!(ax, z, y, permutedims(rho_sorted[k]);
                      colormap = COLORMAP,
                      colorrange = (0.0, vmax))

        if hm_ref === nothing
            hm_ref = hm
        end
    end

    # Barra de color compartida para todas las capturas
    Colorbar(fig[:, 2], hm_ref; label = "ρ_yz")

    savefig(fig, fig_dir, "panel_rho_yz_snapshots")
end

"""Diagnósticos: norma, energía, posiciones medias y referencia z₀+k₀t."""
function save_diagnostics_plot(fig_dir, t_arr, norm_arr, energy_arr,
                               xm_arr, ym_arr, zm_arr)
    fig = Figure(size = (1200, 900))

    ax1 = Axis(fig[1,1]; xlabel = "t (u.a.)", ylabel = "‖Ψ‖²",
               title = "Conservación de la norma")
    lines!(ax1, t_arr, norm_arr; color = :navy, linewidth = 2.0)
    hlines!(ax1, [1.0]; color = (:gray, 0.6), linestyle = :dash)

    ax2 = Axis(fig[1,2]; xlabel = "t (u.a.)", ylabel = "⟨H⟩ (u.a.)",
               title = "Energía cinética")
    lines!(ax2, t_arr, energy_arr; color = :crimson, linewidth = 2.0)
    E_theo = 0.5 * K0^2 + 0.25 * (1/SIGMA_X^2 + 1/SIGMA_Y^2 + 1/SIGMA_Z^2)
    hlines!(ax2, [E_theo]; color = (:gray, 0.6), linestyle = :dash,
            label = @sprintf("teórica = %.3f", E_theo))
    axislegend(ax2; position = :rt)

    ax3 = Axis(fig[2, 1:2]; xlabel = "t (u.a.)", ylabel = "⟨q⟩ (u.a.)",
               title = "Posiciones medias del paquete")
    z_ref = Z0 .+ K0 .* t_arr
    lines!(ax3, t_arr, xm_arr; color = :navy,      linewidth = 2.0, label = "⟨x⟩")
    lines!(ax3, t_arr, ym_arr; color = :darkgreen, linewidth = 2.0, label = "⟨y⟩")
    lines!(ax3, t_arr, zm_arr; color = :crimson,   linewidth = 2.0, label = "⟨z⟩")
    lines!(ax3, t_arr, z_ref;  color = :black,     linewidth = 1.6,
           linestyle = :dash, label = "z₀ + k₀ t")
    axislegend(ax3; position = :lt)

    savefig(fig, fig_dir, "diagnostics_gauss3d")
end

# ════════════════════════════════════════════════════════════════
#  ARCHIVOS DE SALIDA
# ════════════════════════════════════════════════════════════════

function write_run_info(out_dir, dx, dy, dz, Cx, Cy, Cz, Nt,
                        norm0, energy0, snap_actual)
    open(joinpath(out_dir, "run_info.txt"), "w") do io
        println(io, "═"^62)
        println(io, "  $CODE_NAME  v$CODE_VERSION")
        println(io, "  Fecha: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "═"^62)
        println(io)
        println(io, "Ecuación:    i ∂t Ψ = -1/2 ∇² Ψ   (V = 0, ħ = m = 1)")
        println(io, "Método:      Crank-Nicolson ADI 3D, orden alternante")
        println(io, "BC:          Dirichlet 0 en todas las caras")
        println(io)
        @printf(io, "Dominio:     x ∈ [-%.3f, %.3f]   y ∈ [-%.3f, %.3f]   z ∈ [0, %.3f]\n",
                LX/2, LX/2, LY/2, LY/2, LZ)
        @printf(io, "Malla:       NX=%d  NY=%d  NZ=%d\n", NX, NY, NZ)
        @printf(io, "Espaciados:  Δx=%.6f  Δy=%.6f  Δz=%.6f\n", dx, dy, dz)
        println(io)
        @printf(io, "Δt:          %.6e\n", DT_USER)
        @printf(io, "T_final:     %.6e\n", T_FINAL)
        @printf(io, "Nt:          %d\n", Nt)
        println(io)
        println(io, "Coeficientes |ν_d| = ħ Δt /(4 m Δd²):")
        @printf(io, "  Cx = %.6e\n", Cx)
        @printf(io, "  Cy = %.6e\n", Cy)
        @printf(io, "  Cz = %.6e\n", Cz)
        println(io)
        @printf(io, "Centro CI:   (x0, y0, z0) = (%.4f, %.4f, %.4f)\n", X0, Y0, Z0)
        @printf(io, "Anchos:      σx=%.4f  σy=%.4f  σz=%.4f\n",
                SIGMA_X, SIGMA_Y, SIGMA_Z)
        @printf(io, "k0:          %.4f   (vg = k0 = %.4f)\n", K0, K0)
        println(io)
        @printf(io, "Norma inicial:    %.10e\n", norm0)
        @printf(io, "Energía inicial:  %.10e\n", energy0)
        E_theo = 0.5 * K0^2 + 0.25 * (1/SIGMA_X^2 + 1/SIGMA_Y^2 + 1/SIGMA_Z^2)
        @printf(io, "Energía teórica:  %.10e\n", E_theo)
        println(io)
        println(io, "Tiempos de captura (solicitados → efectivos):")
        for (treq, tact) in zip(SNAP_TIMES, snap_actual)
            @printf(io, "  %.6f  →  %.6f\n", treq, tact)
        end
    end
end

function write_diagnostics_dat(data_dir, t_arr, norm_arr,
                               xm_arr, ym_arr, zm_arr,
                               energy_arr, rho_max_arr)
    open(joinpath(data_dir, "diagnostics.dat"), "w") do io
        println(io, "# $CODE_NAME v$CODE_VERSION — diagnostics")
        println(io, "# columnas:")
        println(io, "#  1: t          tiempo (u.a.)")
        println(io, "#  2: norm       ‖Ψ‖²")
        println(io, "#  3: x_mean     ⟨x⟩")
        println(io, "#  4: y_mean     ⟨y⟩")
        println(io, "#  5: z_mean     ⟨z⟩")
        println(io, "#  6: energy     ⟨H⟩")
        println(io, "#  7: rho_yz_max max(ρ_yz(y,z,t))")
        for n in eachindex(t_arr)
            @printf(io, "%.8e %.10e %.8e %.8e %.8e %.10e %.8e\n",
                    t_arr[n], norm_arr[n],
                    xm_arr[n], ym_arr[n], zm_arr[n],
                    energy_arr[n], rho_max_arr[n])
        end
    end
end

# ════════════════════════════════════════════════════════════════
#  ANIMACIÓN OPCIONAL (ρ_yz)
# ════════════════════════════════════════════════════════════════

function make_animation(fig_dir, y, z, anim_frames, anim_times, vmax)
    isempty(anim_frames) && return false
    ext = ANIM_FORMAT == :mp4 ? "mp4" : "gif"
    out_path = joinpath(fig_dir, "rho_yz.$ext")

    fig = Figure(size = (920, 420))
    obs_rho = Observable(permutedims(anim_frames[1]))
    obs_title = Observable(@sprintf("ρ_yz(y, z, t)    t = %.2f", anim_times[1]))
    ax = Axis(fig[1,1];
        xlabel = "z (u.a.)", ylabel = "y (u.a.)",
        title = obs_title,
        aspect = DataAspect(),
    )
    hm = heatmap!(ax, z, y, obs_rho; colormap = COLORMAP, colorrange = (0.0, vmax))
    Colorbar(fig[1, 2], hm; label = "ρ_yz")
    try
        record(fig, out_path, eachindex(anim_frames); framerate = ANIM_FRAMERATE) do k
            obs_rho[]   = permutedims(anim_frames[k])
            obs_title[] = @sprintf("ρ_yz(y, z, t)    t = %.2f", anim_times[k])
        end
        return true
    catch e
        println("    ⚠ Animación falló: ", e)
        return false
    end
end

# ════════════════════════════════════════════════════════════════
#  ENCABEZADO Y RESUMEN EN TERMINAL
# ════════════════════════════════════════════════════════════════

function print_header(out_dir, dx, dy, dz, Cx, Cy, Cz, Nt)
    println("═"^62)
    println("  $CODE_NAME  v$CODE_VERSION")
    println("  Método:   Crank-Nicolson ADI 3D (alternante)")
    println("  Ecuación: i ∂t Ψ = -1/2 ∇² Ψ   (V = 0)")
    println("═"^62)
    println("  Output:   $out_dir")
    @printf("  Dominio:  x ∈ [-%.2f, %.2f]   y ∈ [-%.2f, %.2f]   z ∈ [0, %.2f]\n",
            LX/2, LX/2, LY/2, LY/2, LZ)
    @printf("  Malla:    %d × %d × %d   (Δx=%.5f Δy=%.5f Δz=%.5f)\n",
            NX, NY, NZ, dx, dy, dz)
    @printf("  Tiempo:   Δt=%.3e   T_final=%.3e   Nt=%d\n",
            DT_USER, T_FINAL, Nt)
    @printf("  |ν_d|:    Cx=%.3e   Cy=%.3e   Cz=%.3e\n", Cx, Cy, Cz)
    @printf("  CI:       (x0,y0,z0)=(%.2f,%.2f,%.2f)  σ=(%.2f,%.2f,%.2f)  k0=%.2f\n",
            X0, Y0, Z0, SIGMA_X, SIGMA_Y, SIGMA_Z, K0)
    println("  SNAPS:    ", SNAP_TIMES)
    println("─"^62)
end

function print_progress(n, Nt, t, norm_v, energy_v, t_acum, t_prom, t_est)
    @printf("  [%5d/%-5d]  t=%.4f  ‖Ψ‖²=%.10f  ⟨H⟩=%.6e  t_a=%s  t_est=%s\n",
        n, Nt, t, norm_v, energy_v,
        fmt_time(t_acum), fmt_time(t_est))
end

# ════════════════════════════════════════════════════════════════
#  VALIDACIONES
# ════════════════════════════════════════════════════════════════

function validate_config(dx, dy, dz)
    @assert NX > 5 && NY > 5 && NZ > 5
    @assert DT_USER > 0 && T_FINAL > 0
    @assert -LX/2 ≤ X0 ≤ LX/2 "x0 fuera del dominio"
    @assert -LY/2 ≤ Y0 ≤ LY/2 "y0 fuera del dominio"
    @assert 0 ≤ Z0 ≤ LZ        "z0 fuera del dominio"
    for ts in SNAP_TIMES
        @assert 0 ≤ ts ≤ T_FINAL "SNAP_TIME=$ts fuera de [0, T_final]"
    end
    σ_min = min(SIGMA_X, SIGMA_Y, SIGMA_Z)
    Δ_max = max(dx, dy, dz)
    if σ_min < 2 * Δ_max
        @warn "σ_min=$σ_min < 2·Δ_max=$(2*Δ_max): paquete mal resuelto."
    end
    # Resolución de la longitud de onda de De Broglie λ = 2π/k0
    if K0 > 0
        λ = 2π / K0
        if Δ_max > λ / 6
            @warn "Δ_max=$Δ_max > λ/6=$(λ/6): k0 mal resuelto."
        end
    end
end

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

function main()
    set_thesis_theme!()

    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    out_dir   = ensure_dir(joinpath(OUT_BASE, "run_$timestamp"))
    fig_dir   = ensure_dir(joinpath(out_dir, "figures"))
    data_dir  = ensure_dir(joinpath(out_dir, "data"))

    # ── Malla ──
    x, y, z, dx, dy, dz = build_grid()
    validate_config(dx, dy, dz)
    dV = dx * dy * dz

    # ── CI ──
    psi = gaussian_packet(x, y, z)
    normalize!(psi, dV)

    # ── Coeficientes ν_d (complejos) y matrices CN ──
    nu_x = im * HBAR * DT_USER / (4 * M * dx^2)
    nu_y = im * HBAR * DT_USER / (4 * M * dy^2)
    nu_z = im * HBAR * DT_USER / (4 * M * dz^2)
    Cx, Cy, Cz = abs(nu_x), abs(nu_y), abs(nu_z)
    LU_x = lu(build_LHS(NX, nu_x))
    LU_y = lu(build_LHS(NY, nu_y))
    LU_z = lu(build_LHS(NZ, nu_z))

    # ── Buffers ──
    line_x = Vector{ComplexF64}(undef, NX); rhs_x = similar(line_x)
    line_y = Vector{ComplexF64}(undef, NY); rhs_y = similar(line_y)
    line_z = Vector{ComplexF64}(undef, NZ); rhs_z = similar(line_z)
    lap_buf = Array{ComplexF64,3}(undef, NX, NY, NZ)

    # ── Pasos temporales ──
    Nt = round(Int, T_FINAL / DT_USER)

    # SNAP_TIMES → pasos
    snap_steps  = Int[]
    snap_actual = Float64[]
    for ts in SNAP_TIMES
        n = clamp(round(Int, ts / DT_USER), 0, Nt)
        push!(snap_steps,  n)
        push!(snap_actual, n * DT_USER)
    end

    print_header(out_dir, dx, dy, dz, Cx, Cy, Cz, Nt)

    # ── Diagnósticos por paso ──
    t_arr       = Vector{Float64}(undef, Nt + 1)
    norm_arr    = similar(t_arr)
    energy_arr  = similar(t_arr)
    xm_arr      = similar(t_arr)
    ym_arr      = similar(t_arr)
    zm_arr      = similar(t_arr)
    rho_max_arr = similar(t_arr)

    snap_data = Tuple{Float64, Matrix{Float64}}[]   # (t, ρ_yz)
    anim_frames = Matrix{Float64}[]
    anim_times  = Float64[]

    function record_step!(n_idx, t)
        t_arr[n_idx]      = t
        norm_arr[n_idx]   = norm_squared(psi, dV)
        xm, ym, zm        = expectation_position(psi, x, y, z, dV)
        xm_arr[n_idx]     = xm
        ym_arr[n_idx]     = ym
        zm_arr[n_idx]     = zm
        energy_arr[n_idx] = expectation_energy(psi, lap_buf, dx, dy, dz, dV)
        rho               = rho_yz(psi, dx)
        rho_max_arr[n_idx]= maximum(rho)
        return rho
    end

    function capture_if_needed!(n_step, t, rho)
        if n_step in snap_steps
            push!(snap_data, (t, copy(rho)))
        end
    end

    # ── Estado inicial ──
    rho0 = record_step!(1, 0.0)
    capture_if_needed!(0, 0.0, rho0)
    if GEN_ANIMATION
        push!(anim_frames, copy(rho0)); push!(anim_times, 0.0)
    end
    norm0   = norm_arr[1]
    energy0 = energy_arr[1]

    # ── Loop temporal ──
    println("\nPropagando...")
    t_loop_start = time()
    for n in 1:Nt
        step_CN_ADI!(psi, LU_x, LU_y, LU_z, nu_x, nu_y, nu_z, n,
                     line_x, rhs_x, line_y, rhs_y, line_z, rhs_z)

        t = n * DT_USER
        rho = record_step!(n + 1, t)
        capture_if_needed!(n, t, rho)

        if GEN_ANIMATION && (n % ANIM_EVERY == 0 || n == Nt)
            push!(anim_frames, copy(rho)); push!(anim_times, t)
        end

        if n % PROGRESS_EVERY == 0 || n == Nt
            t_acum = time() - t_loop_start
            t_prom = t_acum / n
            t_est  = t_prom * (Nt - n)
            print_progress(n, Nt, t, norm_arr[n+1], energy_arr[n+1],
                           t_acum, t_prom, t_est)
        end
    end
    t_total = time() - t_loop_start

    # ── Salida: diagnostics.dat ──
    write_diagnostics_dat(data_dir, t_arr, norm_arr,
                          xm_arr, ym_arr, zm_arr,
                          energy_arr, rho_max_arr)

    # ── Figuras ──
    println("\nGenerando figuras...")

    # Escala fija al máximo de ρ_yz en t = 0
    vmax = maximum(snap_data[1][2])
    if vmax <= 0; vmax = maximum(rho_max_arr); end

    # Capturas individuales
    for (t, rho) in snap_data
        save_snapshot_yz(fig_dir, t, y, z, rho, vmax)
    end

    # Panel
    ts_list  = [t  for (t, _)  in snap_data]
    rho_list = [r  for (_, r)  in snap_data]
    save_panel_yz(fig_dir, ts_list, y, z, rho_list, vmax)

    # Diagnósticos
    save_diagnostics_plot(fig_dir, t_arr, norm_arr, energy_arr,
                          xm_arr, ym_arr, zm_arr)

    # Animación (opcional)
    if GEN_ANIMATION
        ok = make_animation(fig_dir, y, z, anim_frames, anim_times, vmax)
        ok && println("  → rho_yz.$(ANIM_FORMAT)")
    end

    # ── run_info.txt ──
    write_run_info(out_dir, dx, dy, dz, Cx, Cy, Cz, Nt,
                   norm0, energy0, snap_actual)

    # ── Resumen final ──
    println("─"^62)
    println("  RESUMEN")
    @printf("    Norma:    min = %.10f   max = %.10f\n",
            minimum(norm_arr), maximum(norm_arr))
    @printf("    Energía:  min = %.6e   max = %.6e\n",
            minimum(energy_arr), maximum(energy_arr))
    E_theo = 0.5 * K0^2 + 0.25 * (1/SIGMA_X^2 + 1/SIGMA_Y^2 + 1/SIGMA_Z^2)
    @printf("    Energía teórica esperada: %.6e\n", E_theo)
    @printf("    ⟨z⟩(T_final) = %.4f   z₀ + k₀ T_final = %.4f\n",
            zm_arr[end], Z0 + K0 * T_FINAL)
    println("    Comparación ⟨z⟩(t)  vs  z₀ + k₀ t  en tiempos de captura:")
    for (treq, tact) in zip(SNAP_TIMES, snap_actual)
        n_idx = clamp(round(Int, tact / DT_USER), 0, Nt) + 1
        @printf("      t=%.3f   ⟨z⟩=%.4f   z₀+k₀t=%.4f   Δ=%.4f\n",
                tact, zm_arr[n_idx], Z0 + K0 * tact,
                zm_arr[n_idx] - (Z0 + K0 * tact))
    end
    @printf("    Coeficientes: Cx=%.3e  Cy=%.3e  Cz=%.3e\n", Cx, Cy, Cz)
    @printf("    Tiempo total: %s\n", fmt_time(t_total))
    println("    Salida:")
    println("      $out_dir/figures/")
    println("      $out_dir/data/")
    println("      $out_dir/run_info.txt")
    println("═"^62)
end

main()
