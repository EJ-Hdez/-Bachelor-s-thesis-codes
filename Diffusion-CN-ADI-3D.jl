# ════════════════════════════════════════════════════════════════
# Diffusion-CN-ADI-3D.jl  v2.0.0
# Difusión 3D (gaussiana) — Crank-Nicolson ADI
#
# Resuelve   ∂t u = D (∂²x + ∂²y + ∂²z) u
# en un dominio cúbico, partiendo de una gaussiana 3D centrada.
# Genera datos crudos y figuras listas para tesis (CairoMakie).
#
# Esquema temporal (un paso completo Δt):
#   ADI: factorización por direcciones (Lie), con orden alternante
#   - Pasos impares: x → y → z
#   - Pasos pares:   z → y → x
#   En cada barrido d se resuelve  A⁺_d u* = A⁻_d u
#   con  A±_d = I ∓ r_d δ²_d  y  r_d = D Δt / (2 Δd²)
#   La alternancia cancela el sesgo direccional a O(Δt).
#
# Notas físico-numéricas:
#   • CN es incondicionalmente estable: no hay condición CFL.
#   • r_x, r_y, r_z se imprimen como diagnóstico de resolución.
#     Valores muy grandes degradan precisión (suavizado), no
#     desestabilizan el método.
#   • BC por defecto: Neumann homogéneo (reflexión virtual).
#     Conserva masa hasta error de máquina/discretización temporal.
#
# Uso:
#   julia Diffusion-CN-ADI-3D.jl
# ════════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Dates
using DelimitedFiles
using CairoMakie

const CODE_NAME    = "Diffusion-CN-ADI-3D.jl"
const CODE_VERSION = "2.0.0"

# ════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DEL USUARIO
# ════════════════════════════════════════════════════════════════

# ── Salida ─────────────────────────────────────────────────────
const OUT_BASE = "/media/usuario/your/path/Diffusion CN"

# ── Dominio y malla ───────────────────────────────────────────
const LX, LY, LZ = 1.0, 1.0, 1.0
const NX, NY, NZ = 121, 121, 121

# ── Parámetros físicos ────────────────────────────────────────
const D_DIFF  = 1.0
const T_FINAL = 0.01
const DT_USER = 7.94e-5 #2e-4

# ── Condición inicial gaussiana ───────────────────────────────
const X0, Y0, Z0 = 0.5, 0.5, 0.5     # centro
const SIGMA0     = 0.10              # ancho inicial
const AMP0       = 1.0               # amplitud
const NORMALIZE_INITIAL_MASS = false # si true, M(0)=1

# ── Numérica ──────────────────────────────────────────────────
# BC_MODE: :neumann0 (recomendada — masa conservada)
#          :dirichlet0 (u=0 en los bordes; masa decae si llega al borde)
const BC_MODE = :neumann0

# ADI_ORDER_MODE: :alternating (impares xyz, pares zyx)
#                 :fixed_xyz   (siempre x→y→z; sesgo O(Δt) visible)
#                 :fixed_zyx   (siempre z→y→x)
const ADI_ORDER_MODE = :alternating

# Reportar valores u<0 (artefacto numérico). Si CLIP_NEGATIVE=true,
# se truncan a cero (sólo si lo activas explícitamente).
const CLIP_NEGATIVE = false

# ── Tiempos de captura (snapshots) ────────────────────────────
const SNAP_TIMES = [0.0, 0.002, 0.004, 0.01]

# ── Animación ─────────────────────────────────────────────────
const GEN_ANIMATION  = true
const ANIM_FORMAT    = :mp4   # :mp4 | :gif
const ANIM_EVERY     = 1      # 1 frame cada N pasos
const ANIM_FRAMERATE = 24

# ── Estilo visual ─────────────────────────────────────────────
const COLORMAP   = :hot
const COLOR_MODE = :global    # :global → misma colorrange en todos los snapshots
const SAVE_PDF   = true
const SAVE_PNG   = true
const DPI        = 300
const FIG_WIDTH  = 800
const FIG_HEIGHT = 700
const FONT_SIZE  = 18
const TITLE_SIZE = 22

# ── Progreso en terminal ──────────────────────────────────────
const PROGRESS_EVERY = 10

# ════════════════════════════════════════════════════════════════
#  TEMA VISUAL (estilo tesis/artículo)
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
            xminorticksvisible = false, yminorticksvisible = false,
            topspinevisible = true, rightspinevisible = true,
        ),
        Legend = (framevisible = false, labelsize = FONT_SIZE),
    ))
end

# ════════════════════════════════════════════════════════════════
#  UTILIDADES
# ════════════════════════════════════════════════════════════════

ensure_dir(p) = (isdir(p) || mkpath(p); p)

function nearest_index(arr, x)
    _, i = findmin(abs.(arr .- x))
    return i
end

# Etiqueta para nombres de archivo: 0.002 → "0p002000"
fmt_t_tag(t) = replace(@sprintf("%.6f", t), "." => "p")

# Guarda figura en PDF y/o PNG
function savefig(fig, dir, name)
    ensure_dir(dir)
    SAVE_PDF && save(joinpath(dir, name * ".pdf"), fig)
    SAVE_PNG && save(joinpath(dir, name * ".png"), fig; px_per_unit = DPI / 96)
end

function fmt_time(s::Float64)
    s < 60 && return @sprintf("%.1fs", s)
    s < 3600 && return @sprintf("%dm%02ds", div(Int(floor(s)),60), mod(Int(floor(s)),60))
    return @sprintf("%dh%02dm", div(Int(floor(s)),3600), div(mod(Int(floor(s)),3600),60))
end

# ════════════════════════════════════════════════════════════════
#  MALLA Y CONDICIÓN INICIAL
# ════════════════════════════════════════════════════════════════

function build_grid()
    x = collect(range(0.0, LX, length = NX))
    y = collect(range(0.0, LY, length = NY))
    z = collect(range(0.0, LZ, length = NZ))
    dx = x[2] - x[1]; dy = y[2] - y[1]; dz = z[2] - z[1]
    return x, y, z, dx, dy, dz
end

function gaussian_initial_condition(x, y, z)
    u = Array{Float64,3}(undef, NX, NY, NZ)
    inv2σ² = 1.0 / (2.0 * SIGMA0^2)
    @inbounds for k in 1:NZ
        dz² = (z[k] - Z0)^2
        for j in 1:NY
            dy² = (y[j] - Y0)^2
            for i in 1:NX
                r² = (x[i] - X0)^2 + dy² + dz²
                u[i,j,k] = AMP0 * exp(-r² * inv2σ²)
            end
        end
    end
    return u
end

function mass(u, dV)
    s = 0.0
    @inbounds for v in u; s += v; end
    return s * dV
end

function normalize_mass!(u, dV)
    M = mass(u, dV)
    M > 0 && (u ./= M)
    return u
end

"""Calcula M, ⟨x⟩, ⟨y⟩, ⟨z⟩ y varianzas σ²_x,y,z en una sola pasada."""
function moments(u, x, y, z, dV)
    Nx, Ny, Nz = size(u)
    M = 0.0
    sx = 0.0; sx2 = 0.0
    sy = 0.0; sy2 = 0.0
    sz = 0.0; sz2 = 0.0
    @inbounds for k in 1:Nz
        zk = z[k]; zk2 = zk*zk
        for j in 1:Ny
            yj = y[j]; yj2 = yj*yj
            for i in 1:Nx
                xi = x[i]; ui = u[i,j,k]
                M  += ui
                sx += xi * ui;  sx2 += xi*xi * ui
                sy += yj * ui;  sy2 += yj2   * ui
                sz += zk * ui;  sz2 += zk2   * ui
            end
        end
    end
    M *= dV
    if M > 0
        invM = 1.0 / M * dV
        mx = sx * invM; my = sy * invM; mz = sz * invM
        var_x = sx2 * invM - mx*mx
        var_y = sy2 * invM - my*my
        var_z = sz2 * invM - mz*mz
    else
        mx = my = mz = 0.0
        var_x = var_y = var_z = 0.0
    end
    return M, mx, my, mz, var_x, var_y, var_z
end

# ════════════════════════════════════════════════════════════════
#  CRANK-NICOLSON 1D + ADI 3D
#
#  Para cada dirección d:
#     A⁺_d u^{n+1} = A⁻_d u^n
#  donde
#     (A⁺_d u)_i = (1+2r) u_i - r u_{i-1} - r u_{i+1}
#     (A⁻_d u)_i = (1-2r) u_i + r u_{i-1} + r u_{i+1}
#
#  Neumann homogéneo: u_{-1}=u_1 ⇒ off-diagonal de borde es ∓2r.
#  Dirichlet 0:        u_{borde}=0 fijo ⇒ ecuación trivial en bordes.
# ════════════════════════════════════════════════════════════════

"""Construye la matriz tridiagonal A⁺ (LHS) según BC."""
function build_LHS(N, r, bc::Symbol)
    diag = fill(1.0 + 2r, N)
    sup  = fill(-r, N - 1)
    sub  = fill(-r, N - 1)
    if bc == :neumann0
        sup[1]   = -2r
        sub[N-1] = -2r
    elseif bc == :dirichlet0
        diag[1] = 1.0; sup[1]   = 0.0
        diag[N] = 1.0; sub[N-1] = 0.0
    else
        error("BC_MODE no soportado: $bc")
    end
    return Tridiagonal(sub, diag, sup)
end

"""Aplica A⁻ (RHS) al vector línea u → out."""
@inline function apply_RHS!(out::Vector{Float64}, u::Vector{Float64}, r::Float64, bc::Symbol)
    N = length(u)
    @inbounds begin
        if bc == :neumann0
            out[1] = (1 - 2r) * u[1] + 2r * u[2]
            for i in 2:N-1
                out[i] = r * u[i-1] + (1 - 2r) * u[i] + r * u[i+1]
            end
            out[N] = 2r * u[N-1] + (1 - 2r) * u[N]
        else  # :dirichlet0
            out[1] = 0.0
            for i in 2:N-1
                out[i] = r * u[i-1] + (1 - 2r) * u[i] + r * u[i+1]
            end
            out[N] = 0.0
        end
    end
end

# Sweeps direccionales: extraen línea, resuelven, devuelven.
function sweep_x!(u, LU_x, r, bc, line, rhs)
    @inbounds for k in 1:NZ, j in 1:NY
        for i in 1:NX; line[i] = u[i,j,k]; end
        apply_RHS!(rhs, line, r, bc)
        ldiv!(LU_x, rhs)
        for i in 1:NX; u[i,j,k] = rhs[i]; end
    end
end

function sweep_y!(u, LU_y, r, bc, line, rhs)
    @inbounds for k in 1:NZ, i in 1:NX
        for j in 1:NY; line[j] = u[i,j,k]; end
        apply_RHS!(rhs, line, r, bc)
        ldiv!(LU_y, rhs)
        for j in 1:NY; u[i,j,k] = rhs[j]; end
    end
end

function sweep_z!(u, LU_z, r, bc, line, rhs)
    @inbounds for j in 1:NY, i in 1:NX
        for k in 1:NZ; line[k] = u[i,j,k]; end
        apply_RHS!(rhs, line, r, bc)
        ldiv!(LU_z, rhs)
        for k in 1:NZ; u[i,j,k] = rhs[k]; end
    end
end

"""Un paso completo CN-ADI con orden alternante."""
function step_CN_ADI!(u, LU_x, LU_y, LU_z, rx, ry, rz, bc, n_step,
                      line_x, rhs_x, line_y, rhs_y, line_z, rhs_z)
    forward = if ADI_ORDER_MODE == :alternating
        isodd(n_step)
    elseif ADI_ORDER_MODE == :fixed_xyz
        true
    else  # :fixed_zyx
        false
    end
    if forward
        sweep_x!(u, LU_x, rx, bc, line_x, rhs_x)
        sweep_y!(u, LU_y, ry, bc, line_y, rhs_y)
        sweep_z!(u, LU_z, rz, bc, line_z, rhs_z)
    else
        sweep_z!(u, LU_z, rz, bc, line_z, rhs_z)
        sweep_y!(u, LU_y, ry, bc, line_y, rhs_y)
        sweep_x!(u, LU_x, rx, bc, line_x, rhs_x)
    end
end

# ════════════════════════════════════════════════════════════════
#  SALIDA: archivos .dat / run_info
# ════════════════════════════════════════════════════════════════

function save_run_info(out_dir, dx, dy, dz, rx, ry, rz, Nt, snap_actual, fig_paths)
    open(joinpath(out_dir, "run_info.txt"), "w") do io
        println(io, "═"^62)
        println(io, "  $CODE_NAME  v$CODE_VERSION")
        println(io, "  Fecha: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "═"^62)
        println(io)
        println(io, "Ecuación:    ∂t u = D (∂²x + ∂²y + ∂²z) u")
        println(io, "Método:      Crank-Nicolson ADI 3D (orden $ADI_ORDER_MODE)")
        println(io, "BC:          $BC_MODE")
        println(io)
        println(io, "Dominio:     [0, $LX] × [0, $LY] × [0, $LZ]")
        @printf(io, "Malla:       NX=%d  NY=%d  NZ=%d\n", NX, NY, NZ)
        @printf(io, "Espaciados:  Δx=%.6f  Δy=%.6f  Δz=%.6f\n", dx, dy, dz)
        println(io)
        @printf(io, "D:           %.6e\n", D_DIFF)
        @printf(io, "Δt:          %.6e\n", DT_USER)
        @printf(io, "T_final:     %.6e\n", T_FINAL)
        @printf(io, "Nt:          %d\n", Nt)
        @printf(io, "r_x:         %.6e\n", rx)
        @printf(io, "r_y:         %.6e\n", ry)
        @printf(io, "r_z:         %.6e\n", rz)
        println(io)
        @printf(io, "Centro CI:   (%.4f, %.4f, %.4f)\n", X0, Y0, Z0)
        @printf(io, "σ₀:          %.6f\n", SIGMA0)
        @printf(io, "Amplitud₀:   %.6e\n", AMP0)
        println(io, "Norm. M(0)=1: ", NORMALIZE_INITIAL_MASS)
        println(io)
        println(io, "SNAP_TIMES (solicitados / efectivos):")
        for (treq, tact) in zip(SNAP_TIMES, snap_actual)
            @printf(io, "  %.6f  →  %.6f\n", treq, tact)
        end
        println(io)
        println(io, "Figuras generadas:")
        for p in fig_paths
            println(io, "  $p")
        end
    end
end

function save_diagnostics(out_dir, t_arr, M_arr, M0, umin_arr, umax_arr,
                          mx_arr, my_arr, mz_arr,
                          vx_arr, vy_arr, vz_arr)
    fname = joinpath(out_dir, "diagnostics.dat")
    open(fname, "w") do io
        println(io, "# $CODE_NAME v$CODE_VERSION — diagnostics")
        println(io, "# Columnas:")
        println(io, "#  1: t            tiempo (a.u. del modelo)")
        println(io, "#  2: mass         masa total ∫u dV")
        println(io, "#  3: rel_mass_err (M(t) - M(0)) / M(0)")
        println(io, "#  4: umin         valor mínimo de u")
        println(io, "#  5: umax         valor máximo de u")
        println(io, "#  6: mean_x       ⟨x⟩")
        println(io, "#  7: mean_y       ⟨y⟩")
        println(io, "#  8: mean_z       ⟨z⟩")
        println(io, "#  9: var_x        σ²_x")
        println(io, "# 10: var_y        σ²_y")
        println(io, "# 11: var_z        σ²_z")
        println(io, "# 12: sigma_x      σ_x")
        println(io, "# 13: sigma_y      σ_y")
        println(io, "# 14: sigma_z      σ_z")
        println(io, "# 15: sigma_anal   √(σ₀² + 2D t)   (referencia, dominio infinito)")
        for n in eachindex(t_arr)
            t  = t_arr[n]
            σa = sqrt(SIGMA0^2 + 2 * D_DIFF * t)
            σx = sqrt(max(vx_arr[n], 0.0))
            σy = sqrt(max(vy_arr[n], 0.0))
            σz = sqrt(max(vz_arr[n], 0.0))
            relerr = M0 != 0 ? (M_arr[n] - M0) / M0 : 0.0
            @printf(io, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                t, M_arr[n], relerr, umin_arr[n], umax_arr[n],
                mx_arr[n], my_arr[n], mz_arr[n],
                vx_arr[n], vy_arr[n], vz_arr[n],
                σx, σy, σz, σa)
        end
    end
end

function save_snapshot_xy(snap_dir, t, x, y, slice_xy)
    tag = fmt_t_tag(t)
    fname = joinpath(snap_dir, "snapshot_t_$(tag)_xy_z0.dat")
    open(fname, "w") do io
        @printf(io, "# t = %.8e   plano xy en z = z₀\n", t)
        println(io, "#   x            y            u(x,y,z0,t)")
        @inbounds for j in 1:NY, i in 1:NX
            @printf(io, "%.8e %.8e %.8e\n", x[i], y[j], slice_xy[i,j])
        end
    end
end

function save_profile_x(snap_dir, t, x, prof_x)
    tag = fmt_t_tag(t)
    fname = joinpath(snap_dir, "profile_x_t_$(tag).dat")
    open(fname, "w") do io
        @printf(io, "# t = %.8e   perfil u(x, y0, z0, t)\n", t)
        println(io, "#   x            u(x,y0,z0,t)")
        @inbounds for i in 1:NX
            @printf(io, "%.8e %.8e\n", x[i], prof_x[i])
        end
    end
end

# ════════════════════════════════════════════════════════════════
#  FIGURAS
# ════════════════════════════════════════════════════════════════

# Color global para que la caída del máximo sea visualmente fiel.
function global_colorrange(slices_xy)
    vmax = -Inf; vmin = Inf
    for S in slices_xy
        @inbounds for v in S
            v > vmax && (vmax = v)
            v < vmin && (vmin = v)
        end
    end
    # Forzar piso en 0 para colormap viridis (clamp visual sin alterar datos).
    return (max(vmin, 0.0), vmax)
end

"""Snapshot individual del plano xy."""
function plot_snapshot(fig_dir, t, x, y, slice_xy, clims)
    fig = Figure(size = (FIG_WIDTH, FIG_HEIGHT))
    ax = Axis(fig[1,1];
        xlabel = "x", ylabel = "y",
        title  = @sprintf("Difusión 3D — plano xy en z = z₀,  t = %.4f", t),
        aspect = DataAspect(),
    )
    hm = heatmap!(ax, x, y, slice_xy; colormap = COLORMAP, colorrange = clims)
    Colorbar(fig[1, 2], hm; label = "u(x,y,z₀,t)")
    savefig(fig, fig_dir, "snap_xy_z0_t_$(fmt_t_tag(t))")
end

"""Panel 2×2 de snapshots con colorbar común."""
function plot_snapshot_panel(fig_dir, snap_t_list, x, y, slices_xy, clims)
    n = length(snap_t_list)
    n == 0 && return
    nrow = n <= 2 ? 1 : 2
    ncol = ceil(Int, n / nrow)

    fig = Figure(size = (480 * ncol + 140, 420 * nrow + 80))
    hm_ref = nothing
    for (k, (t, S)) in enumerate(zip(snap_t_list, slices_xy))
        row = div(k - 1, ncol) + 1
        col = mod(k - 1, ncol) + 1
        ax = Axis(fig[row, col];
            xlabel = row == nrow ? "x" : "",
            ylabel = col == 1    ? "y" : "",
            title  = @sprintf("t = %.4f", t),
            aspect = DataAspect(),
        )
        hm = heatmap!(ax, x, y, S; colormap = COLORMAP, colorrange = clims)
        if hm_ref === nothing; hm_ref = hm; end
    end
    Colorbar(fig[1:nrow, ncol+1], hm_ref; label = "u(x,y,z₀,t)")
    Label(fig[0, 1:ncol+1], "Difusión de una gaussiana 3D — plano xy en z = z₀";
          fontsize = TITLE_SIZE, font = :bold)
    savefig(fig, fig_dir, "panel_xy_z0")
end

"""Heatmap tiempo–x sobre la línea central (y=y0, z=z0)."""
function plot_heatmap_tx(fig_dir, t_arr, x, centerline)
    # centerline: (NX, Nt+1)
    fig = Figure(size = (FIG_WIDTH + 100, FIG_HEIGHT))
    ax = Axis(fig[1,1];
        xlabel = "t", ylabel = "x",
        title  = "Difusión 3D — perfil central u(x, y₀, z₀, t)",
    )
    hm = heatmap!(ax, t_arr, x, permutedims(centerline); colormap = COLORMAP)
    Colorbar(fig[1, 2], hm; label = "u(x,y₀,z₀,t)")
    savefig(fig, fig_dir, "heatmap_tx_centerline")
end

"""Perfiles u(x) en y0,z0 para los tiempos de SNAP_TIMES."""
function plot_profiles(fig_dir, snap_t_list, x, profiles_x)
    fig = Figure(size = (FIG_WIDTH, FIG_HEIGHT))
    ax = Axis(fig[1,1];
        xlabel = "x", ylabel = "u(x, y₀, z₀, t)",
        title  = "Perfil central — caída del máximo y ensanchamiento",
    )
    palette = [:navy, :crimson, :darkgreen, :darkorange, :purple, :teal, :gold]
    for (k, (t, p)) in enumerate(zip(snap_t_list, profiles_x))
        col = palette[mod1(k, length(palette))]
        lines!(ax, x, p; color = col, linewidth = 2.2,
               label = @sprintf("t = %.4f", t))
    end
    axislegend(ax; position = :rt)
    savefig(fig, fig_dir, "profiles_centerline")
end

"""Diagnósticos: masa, máximo y σ_x,y,z(t)."""
function plot_diagnostics(fig_dir, t_arr, M_arr, M0, umax_arr,
                          vx_arr, vy_arr, vz_arr)
    fig = Figure(size = (FIG_WIDTH + 250, 1100))

    # (1) masa relativa
    ax1 = Axis(fig[1,1]; xlabel = "t", ylabel = "M(t) / M(0)",
               title = "Conservación de masa")
    rel = M0 != 0 ? M_arr ./ M0 : zero(M_arr)
    lines!(ax1, t_arr, rel; color = :navy, linewidth = 2.0)
    hlines!(ax1, [1.0]; color = (:gray, 0.6), linestyle = :dash)

    # (2) máximo
    ax2 = Axis(fig[1,2]; xlabel = "t", ylabel = "u_max(t)",
               title = "Caída del máximo")
    lines!(ax2, t_arr, umax_arr; color = :crimson, linewidth = 2.0)

    # (3) σ_x,y,z(t) y σ_analítica
    ax3 = Axis(fig[2, 1:2]; xlabel = "t", ylabel = "σ(t)",
               title = "Ancho efectivo y referencia analítica (dominio infinito)")
    σx = sqrt.(max.(vx_arr, 0.0))
    σy = sqrt.(max.(vy_arr, 0.0))
    σz = sqrt.(max.(vz_arr, 0.0))
    σa = sqrt.(SIGMA0^2 .+ 2 .* D_DIFF .* t_arr)
    lines!(ax3, t_arr, σx; color = :navy,      linewidth = 2.0, label = "σ_x")
    lines!(ax3, t_arr, σy; color = :crimson,   linewidth = 2.0, label = "σ_y")
    lines!(ax3, t_arr, σz; color = :darkgreen, linewidth = 2.0, label = "σ_z")
    lines!(ax3, t_arr, σa; color = :black,     linewidth = 1.6,
           linestyle = :dash, label = "√(σ₀² + 2Dt)")
    axislegend(ax3; position = :lt)

    savefig(fig, fig_dir, "diagnostics_mass_width")
end

"""Animación del plano xy en z = z₀."""
function make_animation(fig_dir, x, y, anim_frames, anim_times, clims)
    isempty(anim_frames) && return false
    ext = ANIM_FORMAT == :mp4 ? "mp4" : "gif"
    out_path = joinpath(fig_dir, "diffusion_xy_z0.$ext")

    fig = Figure(size = (FIG_WIDTH, FIG_HEIGHT))
    obs_u = Observable(anim_frames[1])
    obs_title = Observable(@sprintf("Difusión 3D — plano xy en z = z₀,  t = %.4f", anim_times[1]))
    ax = Axis(fig[1,1]; xlabel = "x", ylabel = "y",
              title = obs_title, aspect = DataAspect())
    hm = heatmap!(ax, x, y, obs_u; colormap = COLORMAP, colorrange = clims)
    Colorbar(fig[1, 2], hm; label = "u(x,y,z₀,t)")

    try
        record(fig, out_path, eachindex(anim_frames); framerate = ANIM_FRAMERATE) do k
            obs_u[]     = anim_frames[k]
            obs_title[] = @sprintf("Difusión 3D — plano xy en z = z₀,  t = %.4f", anim_times[k])
        end
        return true
    catch e
        println("    ⚠ Animación falló (¿ffmpeg disponible?): ", e)
        return false
    end
end

# ════════════════════════════════════════════════════════════════
#  VALIDACIONES
# ════════════════════════════════════════════════════════════════

function validate_config(dx, dy, dz)
    @assert NX > 5 && NY > 5 && NZ > 5  "Malla demasiado pequeña."
    @assert DT_USER > 0                  "DT_USER debe ser > 0."
    @assert T_FINAL > 0                  "T_FINAL debe ser > 0."
    @assert D_DIFF  > 0                  "D_DIFF debe ser > 0."
    @assert 0 ≤ X0 ≤ LX && 0 ≤ Y0 ≤ LY && 0 ≤ Z0 ≤ LZ  "Centro fuera del dominio."
    for ts in SNAP_TIMES
        @assert 0 ≤ ts ≤ T_FINAL "SNAP_TIME=$ts fuera de [0, T_FINAL]."
    end
    σ_min = min(dx, dy, dz)
    if SIGMA0 < 2σ_min
        @warn "σ₀=$SIGMA0 < 2·min(Δ)=$(2σ_min): la gaussiana inicial está mal resuelta."
    end
end

# ════════════════════════════════════════════════════════════════
#  ENCABEZADO Y RESUMEN EN TERMINAL
# ════════════════════════════════════════════════════════════════

function print_header(out_dir, dx, dy, dz, rx, ry, rz, Nt)
    println("═"^62)
    println("  $CODE_NAME  v$CODE_VERSION")
    println("  Método:   Crank-Nicolson ADI 3D ($ADI_ORDER_MODE)")
    println("  Ecuación: ∂t u = D ∇² u")
    println("═"^62)
    println("  Output:   $out_dir")
    @printf("  Dominio:  [0, %.3f] × [0, %.3f] × [0, %.3f]\n", LX, LY, LZ)
    @printf("  Malla:    %d × %d × %d   (Δx=%.5f  Δy=%.5f  Δz=%.5f)\n",
            NX, NY, NZ, dx, dy, dz)
    @printf("  Tiempo:   Δt=%.3e   T_final=%.3e   Nt=%d\n",
            DT_USER, T_FINAL, Nt)
    @printf("  r_d:      r_x=%.3e   r_y=%.3e   r_z=%.3e\n", rx, ry, rz)
    println("  BC:       $BC_MODE")
    rmax = max(rx, ry, rz)
    if rmax > 0.5
        @printf("  ⚠ r_max=%.3f > 0.5: posible suavizado. CN sigue estable, pero la\n", rmax)
        println("    precisión espacio-temporal se degrada (no es inestabilidad).")
    end
    println("  SNAP_TIMES: ", SNAP_TIMES)
    println("─"^62)
end

function print_progress(n, Nt, t, M, M0, umax_v, t_acum, t_prom, t_est)
    @printf("  [%5d/%-5d]  t=%.4f  M/M0=%.6f  u_max=%.4e  t_s=%s  t_a=%s  t_est=%s\n",
        n, Nt, t, (M0 != 0 ? M/M0 : 0.0), umax_v,
        fmt_time(t_prom), fmt_time(t_acum), fmt_time(t_est))
end

function print_summary(M0, M_final, umax0, umax_f, σ0_x, σf_x, σf_y, σf_z, t_total, out_dir)
    println("─"^62)
    println("  RESUMEN")
    @printf("    Masa inicial M(0):       %.8e\n", M0)
    @printf("    Masa final   M(T):       %.8e\n", M_final)
    @printf("    Error relativo de masa:  %.3e\n",
            M0 != 0 ? (M_final - M0)/M0 : 0.0)
    @printf("    u_max(0):                %.6e\n", umax0)
    @printf("    u_max(T):                %.6e\n", umax_f)
    @printf("    σ_x(0)  ≈ σ₀ = %.4f\n", σ0_x)
    @printf("    σ_x(T)  = %.4f   σ_y(T) = %.4f   σ_z(T) = %.4f\n", σf_x, σf_y, σf_z)
    @printf("    σ_analítica(T) = √(σ₀²+2DT) = %.4f\n",
            sqrt(SIGMA0^2 + 2*D_DIFF*T_FINAL))
    @printf("    Tiempo total: %s\n", fmt_time(t_total))
    println("    Salida:       $out_dir")
    println("═"^62)
end

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

function main()
    set_thesis_theme!()

    # ── Salida ──
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    out_dir   = ensure_dir(joinpath(OUT_BASE, "run_$timestamp"))
    fig_dir   = ensure_dir(joinpath(out_dir, "figures"))
    snap_dir  = ensure_dir(joinpath(out_dir, "snapshots"))

    # ── Malla y validación ──
    x, y, z, dx, dy, dz = build_grid()
    validate_config(dx, dy, dz)
    dV = dx * dy * dz

    # ── CI ──
    u = gaussian_initial_condition(x, y, z)
    NORMALIZE_INITIAL_MASS && normalize_mass!(u, dV)

    # ── Coeficientes y matrices CN ──
    rx = D_DIFF * DT_USER / (2 * dx^2)
    ry = D_DIFF * DT_USER / (2 * dy^2)
    rz = D_DIFF * DT_USER / (2 * dz^2)
    LU_x = lu(build_LHS(NX, rx, BC_MODE))
    LU_y = lu(build_LHS(NY, ry, BC_MODE))
    LU_z = lu(build_LHS(NZ, rz, BC_MODE))

    # ── Buffers reutilizables ──
    line_x = Vector{Float64}(undef, NX); rhs_x = similar(line_x)
    line_y = Vector{Float64}(undef, NY); rhs_y = similar(line_y)
    line_z = Vector{Float64}(undef, NZ); rhs_z = similar(line_z)

    # ── Pasos temporales ──
    Nt = round(Int, T_FINAL / DT_USER)

    # Mapear SNAP_TIMES al paso temporal más cercano
    snap_steps  = Int[]
    snap_actual = Float64[]
    for ts in SNAP_TIMES
        n = clamp(round(Int, ts / DT_USER), 0, Nt)
        push!(snap_steps,  n)
        push!(snap_actual, n * DT_USER)
    end

    # Índices del centro para perfiles
    iy0 = nearest_index(y, Y0)
    iz0 = nearest_index(z, Z0)

    print_header(out_dir, dx, dy, dz, rx, ry, rz, Nt)

    # ── Diagnósticos por paso ──
    t_arr   = Vector{Float64}(undef, Nt + 1)
    M_arr   = similar(t_arr); umin_arr = similar(t_arr); umax_arr = similar(t_arr)
    mx_arr  = similar(t_arr); my_arr = similar(t_arr); mz_arr = similar(t_arr)
    vx_arr  = similar(t_arr); vy_arr = similar(t_arr); vz_arr = similar(t_arr)

    centerline = zeros(Float64, NX, Nt + 1)   # u(x, y0, z0, t)

    snap_data_xy = Tuple{Float64, Matrix{Float64}}[]   # (t, slice_xy)
    profile_data = Tuple{Float64, Vector{Float64}}[]   # (t, profile_x)

    anim_frames = Matrix{Float64}[]
    anim_times  = Float64[]

    function record_step!(n_idx, t)
        M, mx, my, mz, vx, vy, vz = moments(u, x, y, z, dV)
        t_arr[n_idx]  = t
        M_arr[n_idx]  = M
        umin_arr[n_idx] = minimum(u)
        umax_arr[n_idx] = maximum(u)
        mx_arr[n_idx], my_arr[n_idx], mz_arr[n_idx] = mx, my, mz
        vx_arr[n_idx], vy_arr[n_idx], vz_arr[n_idx] = vx, vy, vz
        @inbounds for i in 1:NX; centerline[i, n_idx] = u[i, iy0, iz0]; end
        return M, umax_arr[n_idx]
    end

    # ── Estado inicial (n=0) ──
    M0, umax0 = record_step!(1, 0.0)

    # Snapshot/perfil/anim si t=0 está en SNAP_TIMES o si grabamos animación
    function capture_snapshot_if_needed(n_step, t)
        if n_step in snap_steps
            S = u[:, :, iz0]
            push!(snap_data_xy, (t, copy(S)))
            P = u[:, iy0, iz0]
            push!(profile_data, (t, copy(P)))
            save_snapshot_xy(snap_dir, t, x, y, S)
            save_profile_x(snap_dir, t, x, P)
        end
    end
    capture_snapshot_if_needed(0, 0.0)
    if GEN_ANIMATION
        push!(anim_frames, copy(u[:, :, iz0]))
        push!(anim_times,  0.0)
    end

    # ── Loop temporal ──
    t_loop_start = time()
    for n in 1:Nt
        step_CN_ADI!(u, LU_x, LU_y, LU_z, rx, ry, rz, BC_MODE, n,
                     line_x, rhs_x, line_y, rhs_y, line_z, rhs_z)
        CLIP_NEGATIVE && @. u = max(u, 0.0)

        t = n * DT_USER
        M, umax_v = record_step!(n + 1, t)
        capture_snapshot_if_needed(n, t)

        if GEN_ANIMATION && (n % ANIM_EVERY == 0 || n == Nt)
            push!(anim_frames, copy(u[:, :, iz0]))
            push!(anim_times,  t)
        end

        if n % PROGRESS_EVERY == 0 || n == Nt
            t_acum  = time() - t_loop_start
            t_prom  = t_acum / n
            t_est   = t_prom * (Nt - n)
            print_progress(n, Nt, t, M, M0, umax_v, t_acum, t_prom, t_est)
        end
    end
    t_total = time() - t_loop_start

    # ── Diagnósticos a disco ──
    save_diagnostics(out_dir, t_arr, M_arr, M0, umin_arr, umax_arr,
                     mx_arr, my_arr, mz_arr, vx_arr, vy_arr, vz_arr)

    # ── Figuras ──
    println("\nGenerando figuras...")
    fig_paths = String[]

    # Color global para snapshots y animación
    clims = if COLOR_MODE == :global
        all_slices = Matrix{Float64}[]
        append!(all_slices, [s for (_, s) in snap_data_xy])
        GEN_ANIMATION && append!(all_slices, anim_frames)
        global_colorrange(all_slices)
    else
        nothing
    end

    # Panel y snapshots individuales
    if !isempty(snap_data_xy)
        ts_list = [t for (t, _) in snap_data_xy]
        sl_list = [s for (_, s) in snap_data_xy]
        plot_snapshot_panel(fig_dir, ts_list, x, y, sl_list, clims)
        push!(fig_paths, joinpath(fig_dir, "panel_xy_z0.[pdf|png]"))
        for (t, S) in snap_data_xy
            plot_snapshot(fig_dir, t, x, y, S, clims)
            push!(fig_paths, joinpath(fig_dir, "snap_xy_z0_t_$(fmt_t_tag(t)).[pdf|png]"))
        end
    end

    # Heatmap t-x
    plot_heatmap_tx(fig_dir, t_arr, x, centerline)
    push!(fig_paths, joinpath(fig_dir, "heatmap_tx_centerline.[pdf|png]"))

    # Perfiles centrales
    if !isempty(profile_data)
        ts_list  = [t for (t, _) in profile_data]
        prof_list = [p for (_, p) in profile_data]
        plot_profiles(fig_dir, ts_list, x, prof_list)
        push!(fig_paths, joinpath(fig_dir, "profiles_centerline.[pdf|png]"))
    end

    # Diagnósticos
    plot_diagnostics(fig_dir, t_arr, M_arr, M0, umax_arr, vx_arr, vy_arr, vz_arr)
    push!(fig_paths, joinpath(fig_dir, "diagnostics_mass_width.[pdf|png]"))

    # Animación
    if GEN_ANIMATION
        ok = make_animation(fig_dir, x, y, anim_frames, anim_times,
                            clims === nothing ? global_colorrange(anim_frames) : clims)
        ok && push!(fig_paths, joinpath(fig_dir, "diffusion_xy_z0.$(ANIM_FORMAT)"))
    end

    # ── run_info.txt ──
    save_run_info(out_dir, dx, dy, dz, rx, ry, rz, Nt, snap_actual, fig_paths)

    # ── Resumen ──
    σ0_x = sqrt(max(vx_arr[1], 0.0))
    σf_x = sqrt(max(vx_arr[end], 0.0))
    σf_y = sqrt(max(vy_arr[end], 0.0))
    σf_z = sqrt(max(vz_arr[end], 0.0))
    print_summary(M0, M_arr[end], umax0, umax_arr[end],
                  σ0_x, σf_x, σf_y, σf_z, t_total, out_dir)
end

main()
