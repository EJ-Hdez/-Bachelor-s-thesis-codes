# ════════════════════════════════════════════════════════════════
# END-movie.jl  v1.4.0
# Postprocesador gráfico para datos de ENDyn.jl
#
# Compatible con ENDyn.jl v2.2.0
#
# Cambios principales en v1.3.8:
#   • Mantiene compatibilidad con END-movie v1.3.5.
#   • Escala de película: :log, :linear, :both.
#   • Escala lineal con colormap continuo desplazado:
#       blanco en MOVIE_LINEAR_WHITE_VALUE.
#   • Zoom manual opcional en la película yz.
#   • Ticks lineales fijos para parecerse al estilo de Remigio.
# v1.3.8:
#   * Ajuste manual escala log mejorada
# v1.4.0:
#   * Reestructurado generate_nuclear_local_zoom: solo panel local
#     con ventana explícita (H/V) en lugar de semianchos.
#   * Nueva animación generate_movie_traj: trayectorias nucleares
#     locales en tiempo real, sin densidad electrónica.
# ════════════════════════════════════════════════════════════════

using CairoMakie
using Printf
using Dates
using DelimitedFiles
using Statistics
using LinearAlgebra
using GridLayoutBase: Fixed

const CODE_NAME    = "END-movie.jl"
const CODE_VERSION = "1.4.0"

# ════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DEL USUARIO
# ════════════════════════════════════════════════════════════════

# DATA_PATH puede apuntar a:
#   • una simulación individual
#   • un runset completo
#   • una carpeta raíz con múltiples simulaciones
const DATA_PATH = "/media/user/path/of/your/results/from/ENDyn/here"
const OUT_DIR   = "/media/user/new/results"

# ── Tiempo máximo (0 = usar todos los datos) ──────────────────
const T_MAX_MOVIE = 0.0

# ── Qué generar ────────────────────────────────────────────────
const GEN_MOVIE               = false
const GEN_MOVIE_TRAJ          = false
const GEN_SNAPSHOTS           = false
const GEN_SNAPSHOT_PANEL      = false
const GEN_TRAJECTORY          = false
const GEN_ENERGY_PLOTS        = false
const GEN_DENSITY_FINAL       = false
const GEN_MOMENTUM            = true
const GEN_COLLISION_PLANE     = false
const GEN_ANGULAR_MOMENTUM    = false
const GEN_INITIAL_ATOM_REGION = false
const GEN_NUCLEAR_LOCAL_ZOOM  = false

# ── Zoom nuclear local y animación de trayectorias ─────────────
# Ventana visual explícita en notación NEUTRA (no asume z–y):
#   H = eje horizontal del gráfico = eje de colisión (dir_collision)
#   V = eje vertical   del gráfico = eje del parámetro de impacto (dir_b)
#
# :auto   → ventana automática derivada del rango efectivo de la
#           trayectoria del blanco dentro de T_MAX_MOVIE, con padding.
# :manual → usa NUCLEAR_LOCAL_H_AXIS y NUCLEAR_LOCAL_V_AXIS.
const NUCLEAR_LOCAL_VIEW_MODE = :manual
const NUCLEAR_LOCAL_H_AXIS    = (-1.0, 1.0)
const NUCLEAR_LOCAL_V_AXIS    = (-1.0, 1.0)

# Guías cruzadas en el blanco inicial (solo panel estático)
const NUCLEAR_LOCAL_SHOW_GUIDES = true
const NUCLEAR_LOCAL_GUIDE_COLOR = (:black, 0.35)

# Estilo de líneas y marcadores nucleares (panel estático y animación)
const NUCLEAR_LOCAL_TARGET_COLOR = :blue
const NUCLEAR_LOCAL_PROJ_COLOR   = :orange
const NUCLEAR_LOCAL_TARGET_LW    = 1.0
const NUCLEAR_LOCAL_PROJ_LW      = 1.0

# Tamaño del marcador de "posición actual" en la animación
const NUCLEAR_LOCAL_NOW_MARKERSIZE = 4.0

# ── Plano de colisión ──────────────────────────────────────────
const CPLANE_COLOR_TARGET = :blue
const CPLANE_COLOR_PROJ   = :orange
const CPLANE_COLOR_CM     = :black
const CPLANE_ZERO_COLOR   = (:gray, 0.5)

# ── Región fija alrededor del átomo inicial ───────────────────
const INITIAL_REGION_HALF_Z = 5.0
const INITIAL_REGION_HALF_Y = 5.0

const INITIAL_REGION_SHOW_GUIDES = true
const INITIAL_REGION_GUIDE_COLOR = (:black, 0.35)

# ── Formato de salida ──────────────────────────────────────────
const SAVE_PDF = true
const SAVE_PNG = false
const PX_PER_UNIT = 2

# ── Película / snapshots yz ────────────────────────────────────
const MOVIE_FPS      = 15
const MOVIE_COLORMAP = :bwr

# ── Escala logarítmica ─────────────────────────────────────────
# :auto   → estima máximo real y abre ventana log (no funciona muy bien)
# :manual → usa límites fijados abajo
const MOVIE_LOG_MODE         = :manual
const MOVIE_LOG_SPAN_DECADES = 3.5 #Significa cuántas décadas por debajo del máximo quieres mostrar
const MOVIE_LOG_FLOOR_MANUAL = -4.0
const MOVIE_LOG_CEIL_MANUAL  = -0.5
const MOVIE_LOG_MIN_FLOOR    = -5.0
# Valor donde cae el blanco en la escala logarítmica.
# OJO: este valor está en unidades log10, no en densidad lineal.
# Ejemplos:
#   MOVIE_LOG_WHITE_VALUE = -2.0  → blanco en 10^-2 = 0.01
#   MOVIE_LOG_WHITE_VALUE = -1.0  → blanco en 10^-1 = 0.1
const MOVIE_LOG_WHITE_VALUE = -2.5

# ── Escala de color global ─────────────────────────────────────
# :log, :linear, :both
const MOVIE_SCALE = :linear

# ── Escala lineal ──────────────────────────────────────────────
# :auto   → usa el máximo real de los snapshots
# :manual → usa MOVIE_LINEAR_CEIL_MANUAL
const MOVIE_LINEAR_MODE        = :manual
const MOVIE_LINEAR_CEIL_MANUAL = 0.10
const MOVIE_LINEAR_CEIL_FRAC   = 1.0

# Valor lineal que se mostrará como BLANCO en la barra y el heatmap.
# Con clims = (0, 0.1), si pones 0.01:
#   azul → 0
#   blanco → 0.01
#   rojo → 0.10
const MOVIE_LINEAR_WHITE_VALUE = 0.01

# Etiqueta y ticks lineales
const MOVIE_LINEAR_SHOW_LABEL  = true
const MOVIE_LINEAR_TICKS       = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

# ── Ventana visible de la película ──────────────────────────
# :full   → muestra todo el dominio
# :manual → usa MOVIE_ZLIMS y MOVIE_YLIMS
const MOVIE_VIEW_MODE = :manual
const MOVIE_ZLIMS = (-10.0, 10.0)    
const MOVIE_YLIMS = (-10.0, 10.0)

# ── Fotocapturas ───────────────────────────────────────────────
const SNAP_TIMES  = [42.0, 114.0, 136.0, 142.0, 160.0, 250.0]
const SNAP_FORMAT = :png   # :png o :pdf

# ── Panel de snapshots ─────────────────────────────────────────
const PANEL_ROWS = 3
const PANEL_COLS = 2

# ── Densidad final ρ(z) ────────────────────────────────────────
const SHOW_OMEGA_LINE   = false
const OMEGA_LINE_COLOR  = :gray
const DENSITY_COLOR     = :black
const OMEGA_TEXT_DX     = 0.5
const OMEGA_TEXT_Y_FRAC = 0.90

# ── Marcadores nucleares Unicode ───────────────────────────────
# target = ⊗, proyectil = ⊕, centro de masa = +
const NUCLEAR_FONT  = "Noto Sans Symbols 2"
const SYMBOL_TARGET = '⊗'
const SYMBOL_PROJ   = '⊕'
const SYMBOL_CM     = '+'

const SIZE_TARGET = 15
const SIZE_PROJ   = 15
const SIZE_CM     = 12

const COLOR_TARGET = :black
const COLOR_PROJ   = :black
const COLOR_CM     = :black

# Ajuste fino visual de glifos
const TARGET_DZ = 0.0
const TARGET_DY = 0.0
const PROJ_DZ   = 0.0
const PROJ_DY   = 0.0
const CM_DZ     = 0.0
const CM_DY     = 0.0

# ── Trayectorias 3D ────────────────────────────────────────────
const TRAJ_IMPACT_ASPECT       = 0.58
const TRAJ_COLLISION_ASPECT    = 1.10
const TRAJ_TRANSVERSE_PAD      = 0.35
const TRAJ_COLLISION_PAD       = 0.08
const TRAJ_MIN_TRANSVERSE_SPAN = 1.8
const TRAJ_MIN_COLLISION_SPAN  = 8.0
const TRAJ_INFO_FONT_REDUCTION = 4

# ── Estilo ─────────────────────────────────────────────────────
const FONT_SIZE  = 18
const TITLE_SIZE = 22

# ════════════════════════════════════════════════════════════════
#  LECTURA DE DATOS
# ════════════════════════════════════════════════════════════════

function read_snapshot(filepath::String)
    t_val = NaN
    R1   = [NaN, NaN, NaN]
    R2   = [NaN, NaN, NaN]
    Rcm  = [NaN, NaN, NaN]
    data_lines = String[]

    open(filepath, "r") do io
        for line in eachline(io)
            if startswith(line, "# t = ")
                t_val = parse(Float64, split(line, "=")[2])

            elseif startswith(line, "# R1 = ")
                vals = split(strip(split(line, "=", limit = 2)[2]))
                R1 = [parse(Float64, v) for v in vals]

            elseif startswith(line, "# R2 = ")
                vals = split(strip(split(line, "=", limit = 2)[2]))
                R2 = [parse(Float64, v) for v in vals]

            elseif startswith(line, "# Rcm = ")
                vals = split(strip(split(line, "=", limit = 2)[2]))
                Rcm = [parse(Float64, v) for v in vals]

            elseif !startswith(line, "#")
                push!(data_lines, line)
            end
        end
    end

    y_v = Float64[]
    z_v = Float64[]
    r_v = Float64[]

    for line in data_lines
        p = split(line)
        length(p) >= 3 || continue
        push!(y_v, parse(Float64, p[1]))
        push!(z_v, parse(Float64, p[2]))
        push!(r_v, parse(Float64, p[3]))
    end

    yu = sort(unique(y_v))
    zu = sort(unique(z_v))
    Ny = length(yu)
    Nz = length(zu)

    rho = zeros(Ny, Nz)

    for k in eachindex(y_v)
        iy = searchsortedfirst(yu, y_v[k])
        iz = searchsortedfirst(zu, z_v[k])
        if iy <= Ny && iz <= Nz
            rho[iy, iz] = r_v[k]
        end
    end

    return (t = t_val, R1 = R1, R2 = R2, Rcm = Rcm, y = yu, z = zu, rho = rho)
end

function read_traj_log(filepath::String)
    data = readdlm(filepath, comments = true)
    return (
        t   = data[:, 1],
        R1x = data[:, 2],  R1y = data[:, 3],  R1z = data[:, 4],
        R2x = data[:, 5],  R2y = data[:, 6],  R2z = data[:, 7],
        P1x = data[:, 8],  P1y = data[:, 9],  P1z = data[:,10],
        P2x = data[:,11],  P2y = data[:,12],  P2z = data[:,13],
        CMx = data[:,14],  CMy = data[:,15],  CMz = data[:,16],
    )
end

function read_energy_log(filepath::String)
    data = readdlm(filepath, comments = true)
    ncol = size(data, 2)

    col_or_nan(j) = j <= ncol ? data[:, j] : fill(NaN, size(data, 1))

    return (
        t         = data[:, 1],
        K_T       = col_or_nan(2),
        K_P       = col_or_nan(3),
        V_nn      = col_or_nan(4),
        T_e       = col_or_nan(5),
        V_eN      = col_or_nan(6),
        E_elec    = col_or_nan(7),
        E_total   = col_or_nan(8),
        norm      = col_or_nan(9),
        T_e_n     = col_or_nan(10),
        V_eN_n    = col_or_nan(11),
        E_elec_n  = col_or_nan(12),
        E_total_n = col_or_nan(13),
    )
end

function read_density_z(filepath::String)
    data = readdlm(filepath, comments = true)
    return (z = data[:, 1], rho = data[:, 2])
end

function read_momentum_kz(filepath::String)
    data = readdlm(filepath, comments = true)
    return (kz = data[:, 1], rho = data[:, 2])
end

function read_run_info(filepath::String)
    info = Dict{String,String}()
    isfile(filepath) || return info

    for line in readlines(filepath)
        if contains(line, ": ") && !startswith(line, "─")
            parts = split(line, ": ", limit = 2)
            length(parts) == 2 || continue
            info[strip(parts[1])] = strip(parts[2])
        end
    end

    return info
end

function info_get(info::Dict{String,String}, key::String, default::String = "?")
    haskey(info, key) && return info[key]
    for (k, v) in info
        occursin(key, k) && return v
    end
    return default
end

function find_snapshots(sim_dir::String)
    snap_dir = joinpath(sim_dir, "datos_dens_yz")
    isdir(snap_dir) || return String[]

    files = filter(f -> startswith(f, "dens_yz_") && endswith(f, ".dat"),
                   readdir(snap_dir))
    sort!(files)
    return [joinpath(snap_dir, f) for f in files]
end

function time_mask(t_array, tmax)
    return tmax > 0.0 ? findall(t -> t <= tmax, t_array) : collect(1:length(t_array))
end

# ════════════════════════════════════════════════════════════════
#  UTILIDADES GEOMÉTRICAS
# ════════════════════════════════════════════════════════════════

function remaining_axis(a::Symbol, b::Symbol)
    for ax in (:x, :y, :z)
        (ax != a && ax != b) && return ax
    end
    return :x
end

function axis_label_str(ax::Symbol)
    ax == :x && return "x"
    ax == :y && return "y"
    return "z"
end

function expected_normal_axis(dir_collision::Symbol, dir_b::Symbol)
    return remaining_axis(dir_collision, dir_b)
end

function unit_or_nan(x, y, z)
    n = sqrt(x*x + y*y + z*z)
    if !isfinite(n) || n <= 1e-30
        return (NaN, NaN, NaN, NaN)
    end
    return (x / n, y / n, z / n, n)
end

function traj_coord(tr, who::Symbol, ax::Symbol)
    if who == :target
        ax == :x && return tr.R1x
        ax == :y && return tr.R1y
        return tr.R1z

    elseif who == :proj
        ax == :x && return tr.R2x
        ax == :y && return tr.R2y
        return tr.R2z

    elseif who == :cm
        ax == :x && return tr.CMx
        ax == :y && return tr.CMy
        return tr.CMz
    end

    error("who debe ser :target, :proj o :cm")
end

# ════════════════════════════════════════════════════════════════
#  UTILIDADES VISUALES
# ════════════════════════════════════════════════════════════════

function sup_str(n::Int)
    sups = Dict(
        '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
        '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹',
        '-' => '⁻'
    )
    return join(get(sups, c, c) for c in string(n))
end

function log10_ticks(lo::Float64, hi::Float64)
    vals   = collect(ceil(Int, lo):floor(Int, hi))
    fvals  = Float64.(vals)
    labels = ["10" * sup_str(v) for v in vals]
    return (fvals, labels)
end

function safe_positive_max(A)
    m = 0.0
    @inbounds for x in A
        if isfinite(x) && x > m
            m = x
        end
    end
    return m
end

function max_rho_in_snapshot_file(filepath::String)
    m = 0.0
    open(filepath, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            p = split(line)
            length(p) >= 3 || continue
            ρ = tryparse(Float64, p[3])
            if ρ !== nothing && isfinite(ρ) && ρ > m
                m = ρ
            end
        end
    end
    return m
end

function estimate_log_clims(snapshot_files::Vector{String};
                            extra_files::Vector{String} = String[])
    if MOVIE_LOG_MODE == :manual
        return (MOVIE_LOG_FLOOR_MANUAL, MOVIE_LOG_CEIL_MANUAL)
    end

    maxrho = 0.0

    for f in snapshot_files
        maxrho = max(maxrho, max_rho_in_snapshot_file(f))
    end

    for f in extra_files
        isfile(f) || continue
        maxrho = max(maxrho, max_rho_in_snapshot_file(f))
    end

    maxrho = max(maxrho, 1e-12)
    hi = ceil(log10(maxrho) * 10) / 10
    lo = max(MOVIE_LOG_MIN_FLOOR, hi - MOVIE_LOG_SPAN_DECADES)

    return (lo, hi)
end

function snapshot_to_logrho(rho::AbstractMatrix, clims::Tuple{Float64,Float64})
    lo, _ = clims
    ρfloor = 10.0^lo
    return log10.(max.(rho, ρfloor))
end

function estimate_linear_clims(snapshot_files::Vector{String};
                               extra_files::Vector{String} = String[])
    if MOVIE_LINEAR_MODE == :manual
        return (0.0, MOVIE_LINEAR_CEIL_MANUAL)
    end

    maxrho = 0.0

    for f in snapshot_files
        maxrho = max(maxrho, max_rho_in_snapshot_file(f))
    end

    for f in extra_files
        isfile(f) || continue
        maxrho = max(maxrho, max_rho_in_snapshot_file(f))
    end

    maxrho = max(maxrho, 1e-12)
    hi = maxrho * MOVIE_LINEAR_CEIL_FRAC

    return (0.0, hi)
end

function snapshot_to_linrho(rho::AbstractMatrix, clims::Tuple{Float64,Float64})
    lo, hi = clims
    return clamp.(rho, lo, hi)
end

function build_shifted_linear_colormap(clims::Tuple{Float64,Float64})
    lo, hi = clims
    hi > lo || error("clims inválido en build_shifted_linear_colormap")

    w = (MOVIE_LINEAR_WHITE_VALUE - lo) / (hi - lo)
    w = clamp(w, 1e-6, 1.0 - 1e-6)

    return Makie.cgrad(
        [:blue, :white, :red],
        [0.0, w, 1.0]
    )
end

function build_linear_ticks(clims::Tuple{Float64,Float64})
    lo, hi = clims
    vals = [v for v in MOVIE_LINEAR_TICKS if lo - 1e-12 <= v <= hi + 1e-12]
    labels = String[]
    for v in vals
        if isapprox(v, 0.0; atol = 1e-12)
            push!(labels, "0")
        elseif isapprox(v, 0.1; atol = 1e-12)
            push!(labels, "0.1")
        else
            push!(labels, @sprintf("%.2f", v))
        end
    end
    return (vals, labels)
end

function build_shifted_log_colormap(clims::Tuple{Float64,Float64})
    lo, hi = clims
    hi > lo || error("clims inválido en build_shifted_log_colormap")

    w = (MOVIE_LOG_WHITE_VALUE - lo) / (hi - lo)
    w = clamp(w, 1e-6, 1.0 - 1e-6)

    return Makie.cgrad(
        [:blue, :white, :red],
        [0.0, w, 1.0]
    )
end

function yz_plot_box_size(; plot_height::Int = 360, cb_width::Int = 28)
    if MOVIE_VIEW_MODE == :manual
        zspan = max(MOVIE_ZLIMS[2] - MOVIE_ZLIMS[1], 1e-12)
        yspan = max(MOVIE_YLIMS[2] - MOVIE_YLIMS[1], 1e-12)
        plot_width = Int(round(plot_height * zspan / yspan))
    else
        plot_width = Int(round(1.45 * plot_height))
    end

    fig_width  = plot_width + cb_width + 130
    fig_height = plot_height + 150
    return fig_width, fig_height, plot_width, plot_height, cb_width
end

function apply_movie_view!(ax)
    if MOVIE_VIEW_MODE == :manual
        xlims!(ax, MOVIE_ZLIMS[1], MOVIE_ZLIMS[2])
        ylims!(ax, MOVIE_YLIMS[1], MOVIE_YLIMS[2])
    end
end

function prepare_yz_scale(scale::Symbol, snap_files::Vector{String}, final_yz::String)
    if scale == :log
        clims = estimate_log_clims(snap_files; extra_files = [final_yz])
        transform = (ρ -> snapshot_to_logrho(ρ, clims))
        hm_cmap = build_shifted_log_colormap(clims)

        tvals, tlabels = log10_ticks(clims[1], clims[2])
        cb_label = rich("log", subscript("10"), " ρ(y,z,t)")
        cb_ticks = (tvals, tlabels)

        file_tag  = "log"
        human_tag = "log"

    else
        clims = estimate_linear_clims(snap_files; extra_files = [final_yz])
        transform = (ρ -> snapshot_to_linrho(ρ, clims))
        hm_cmap = build_shifted_linear_colormap(clims)

        cb_label = MOVIE_LINEAR_SHOW_LABEL ? "ρ(y,z,t)" : ""
        cb_ticks = build_linear_ticks(clims)

        file_tag  = "linear"
        human_tag = "lineal"
    end

    return clims, transform, hm_cmap, cb_label, cb_ticks, file_tag, human_tag
end

# ════════════════════════════════════════════════════════════════
#  ESTILO
# ════════════════════════════════════════════════════════════════

function set_science_theme!()
    set_theme!(Theme(
        fontsize = FONT_SIZE,
        fonts = (; regular = "TeX Gyre Termes", bold = "TeX Gyre Termes Bold"),

        Axis = (
            xgridvisible = false,
            ygridvisible = false,
            xlabelsize = FONT_SIZE + 2,
            ylabelsize = FONT_SIZE + 2,
            xticklabelsize = FONT_SIZE - 4,
            yticklabelsize = FONT_SIZE - 4,
            titlesize = TITLE_SIZE,
            spinewidth = 1.5,
            xtickwidth = 1.2,
            ytickwidth = 1.2,
            topspinevisible = true,
            rightspinevisible = true,
        ),

        Legend = (
            framevisible = false,
            labelsize = FONT_SIZE - 2,
        ),
    ))
end

# ════════════════════════════════════════════════════════════════
#  GUARDADO DE FIGURAS
# ════════════════════════════════════════════════════════════════

function save_fig_base(basepath::String, fig)
    SAVE_PDF && save(basepath * ".pdf", fig; px_per_unit = PX_PER_UNIT)
    SAVE_PNG && save(basepath * ".png", fig; px_per_unit = PX_PER_UNIT)
end

# ════════════════════════════════════════════════════════════════
#  MARCADORES NUCLEARES UNICODE
# ════════════════════════════════════════════════════════════════

function draw_markers!(ax, R1, R2, Rcm)
    if !any(isnan, R1)
        scatter!(ax, [R1[3] + TARGET_DZ], [R1[2] + TARGET_DY];
            marker = SYMBOL_TARGET,
            markersize = SIZE_TARGET,
            color = COLOR_TARGET,
            font = NUCLEAR_FONT)
    end

    if !any(isnan, R2)
        scatter!(ax, [R2[3] + PROJ_DZ], [R2[2] + PROJ_DY];
            marker = SYMBOL_PROJ,
            markersize = SIZE_PROJ,
            color = COLOR_PROJ,
            font = NUCLEAR_FONT)
    end

    if !any(isnan, Rcm)
        scatter!(ax, [Rcm[3] + CM_DZ], [Rcm[2] + CM_DY];
            marker = SYMBOL_CM,
            markersize = SIZE_CM,
            color = COLOR_CM,
            font = NUCLEAR_FONT)
    end
end

# ════════════════════════════════════════════════════════════════
#  PELÍCULA MP4
# ════════════════════════════════════════════════════════════════

function generate_movie(sim_dir::String, out_dir::String)
    snap_files = find_snapshots(sim_dir)
    isempty(snap_files) && (println("  ⚠ Sin snapshots"); return)

    info  = read_run_info(joinpath(sim_dir, "run_info.txt"))
    E_str = info_get(info, "E_keV")
    b_str = info_get(info, "b")

    if T_MAX_MOVIE > 0.0
        filtered = String[]
        for f in snap_files
            snap = read_snapshot(f)
            snap.t <= T_MAX_MOVIE && push!(filtered, f)
        end
        snap_files = filtered
    end

    isempty(snap_files) && (println("  ⚠ Sin snapshots en rango"); return)

    final_yz = joinpath(sim_dir, "dens_final_yz.dat")

    scales = MOVIE_SCALE == :both   ? [:log, :linear] :
             MOVIE_SCALE == :linear ? [:linear] :
                                      [:log]

    for scale in scales
        _record_movie(snap_files, final_yz, sim_dir, out_dir, E_str, b_str, scale)
    end
end

function _record_movie(snap_files, final_yz, sim_dir, out_dir,
                       E_str, b_str, scale::Symbol)

    clims, transform, hm_cmap, cb_label, cb_ticks, file_tag, human_tag =
        prepare_yz_scale(scale, snap_files, final_yz)

    println("  Generando película [$human_tag] ($(length(snap_files)) frames)...")

    snap0 = read_snapshot(snap_files[1])
    rho0  = transform(snap0.rho)

    fig_w, fig_h, plot_w, plot_h, cb_w = yz_plot_box_size(plot_height = 380, cb_width = 30)
    fig = Figure(size = (fig_w, fig_h))

    title_obs = Observable(
        @sprintf("E = %s keV, b = %s  |  t = %.1f a.u.", E_str, b_str, snap0.t)
    )

    ax = Axis(fig[1, 1];
        xlabel = "z (a.u.)",
        ylabel = "y (a.u.)",
        title  = title_obs,
        aspect = DataAspect()
    )

    apply_movie_view!(ax)

    rho_obs = Observable(rho0')

    Tz = Observable([snap0.R1[3]  + TARGET_DZ])
    Ty = Observable([snap0.R1[2]  + TARGET_DY])
    Pz = Observable([snap0.R2[3]  + PROJ_DZ])
    Py = Observable([snap0.R2[2]  + PROJ_DY])
    Cz = Observable([snap0.Rcm[3] + CM_DZ])
    Cy = Observable([snap0.Rcm[2] + CM_DY])

    heatmap!(ax, snap0.z, snap0.y, rho_obs;
        colormap   = hm_cmap,
        colorrange = clims
    )

    scatter!(ax, Tz, Ty;
        marker = SYMBOL_TARGET,
        markersize = SIZE_TARGET,
        color = COLOR_TARGET,
        font = NUCLEAR_FONT
    )

    scatter!(ax, Pz, Py;
        marker = SYMBOL_PROJ,
        markersize = SIZE_PROJ,
        color = COLOR_PROJ,
        font = NUCLEAR_FONT
    )

    scatter!(ax, Cz, Cy;
        marker = SYMBOL_CM,
        markersize = SIZE_CM,
        color = COLOR_CM,
        font = NUCLEAR_FONT
    )

    Colorbar(fig[1, 2];
        colormap   = hm_cmap,
        colorrange = clims,
        label      = cb_label,
        ticks      = cb_ticks
    )
    colgap!(fig.layout, 0)

    rowsize!(fig.layout, 1, Fixed(plot_h))
    colsize!(fig.layout, 1, Fixed(plot_w))
    colsize!(fig.layout, 2, Fixed(cb_w))

    movie_path = joinpath(out_dir, "dens_yz_movie_" * file_tag * ".mp4")

    record(fig, movie_path, 1:length(snap_files); framerate = MOVIE_FPS) do i
        snap = read_snapshot(snap_files[i])

        rho_obs[]   = transform(snap.rho)'
        title_obs[] = @sprintf("E = %s keV, b = %s  |  t = %.1f a.u.",
                               E_str, b_str, snap.t)

        Tz[] = [snap.R1[3]  + TARGET_DZ]
        Ty[] = [snap.R1[2]  + TARGET_DY]
        Pz[] = [snap.R2[3]  + PROJ_DZ]
        Py[] = [snap.R2[2]  + PROJ_DY]
        Cz[] = [snap.Rcm[3] + CM_DZ]
        Cy[] = [snap.Rcm[2] + CM_DY]
    end

    empty!(fig)
    GC.gc(true)
    println("  ✓ Película [$human_tag]: $movie_path")
end

# ════════════════════════════════════════════════════════════════
#  FOTOCAPTURAS
# ════════════════════════════════════════════════════════════════

function plot_snapshot(snap, clims, transform, hm_cmap, cb_label, cb_ticks; title_str = "")
    fig_w, fig_h, plot_w, plot_h, cb_w = yz_plot_box_size(plot_height = 330, cb_width = 28)
    fig = Figure(size = (fig_w, fig_h))

    ax = Axis(fig[1, 1];
        xlabel = "z (a.u.)",
        ylabel = "y (a.u.)",
        title  = title_str,
        aspect = DataAspect()
    )

    apply_movie_view!(ax)

    rho_plot = transform(snap.rho)

    heatmap!(ax, snap.z, snap.y, rho_plot';
        colormap   = hm_cmap,
        colorrange = clims
    )

    draw_markers!(ax, snap.R1, snap.R2, snap.Rcm)

    Colorbar(fig[1, 2];
        colormap   = hm_cmap,
        colorrange = clims,
        label      = cb_label,
        ticks      = cb_ticks
    )
    colgap!(fig.layout, 0)

    rowsize!(fig.layout, 1, Fixed(plot_h))
    colsize!(fig.layout, 1, Fixed(plot_w))
    colsize!(fig.layout, 2, Fixed(cb_w))

    return fig
end

function generate_snapshots(sim_dir::String, out_dir::String)
    snap_files = find_snapshots(sim_dir)
    isempty(snap_files) && return

    all_snaps = [(f, read_snapshot(f)) for f in snap_files]
    ts = [s[2].t for s in all_snaps]

    final_yz = joinpath(sim_dir, "dens_final_yz.dat")
    info = read_run_info(joinpath(sim_dir, "run_info.txt"))
    E_str = info_get(info, "E_keV")
    b_str = info_get(info, "b")

    scales = MOVIE_SCALE == :both   ? [:log, :linear] :
             MOVIE_SCALE == :linear ? [:linear] :
                                      [:log]

    println("  Generando fotocapturas...")

    for scale in scales
        clims, transform, hm_cmap, cb_label, cb_ticks, file_tag, human_tag =
            prepare_yz_scale(scale, snap_files, final_yz)

        for tref in SNAP_TIMES
            T_MAX_MOVIE > 0.0 && tref > T_MAX_MOVIE && continue

            _, i = findmin(abs.(ts .- tref))
            snap = all_snaps[i][2]

            fig = plot_snapshot(
                snap,
                clims,
                transform,
                hm_cmap,
                cb_label,
                cb_ticks;
                title_str = @sprintf("E = %s keV, b = %s  |  t = %.1f a.u.",
                                     E_str, b_str, snap.t)
            )

            stem = @sprintf("snapshot_t%.1f_%s", snap.t, file_tag)

            if SNAP_FORMAT == :pdf
                save(joinpath(out_dir, stem * ".pdf"), fig; px_per_unit = PX_PER_UNIT)
            else
                save(joinpath(out_dir, stem * ".png"), fig; px_per_unit = PX_PER_UNIT)
            end

            empty!(fig)
            GC.gc(true)
        end
    end

    println(" ✓ Fotocapturas generadas")
end

# ════════════════════════════════════════════════════════════════
#  PANEL DE SNAPSHOTS
# ════════════════════════════════════════════════════════════════

function generate_snapshot_panel(sim_dir::String, out_dir::String)
    snap_files = find_snapshots(sim_dir)
    isempty(snap_files) && return

    all_snaps = [(f, read_snapshot(f)) for f in snap_files]
    ts = [s[2].t for s in all_snaps]

    use_times = T_MAX_MOVIE > 0 ?
        filter(t -> t <= T_MAX_MOVIE, SNAP_TIMES) :
        copy(SNAP_TIMES)

    np = min(PANEL_ROWS * PANEL_COLS, length(use_times))
    np < 1 && return
    use_times = use_times[1:np]

    final_yz = joinpath(sim_dir, "dens_final_yz.dat")

    scales = MOVIE_SCALE == :both   ? [:log, :linear] :
             MOVIE_SCALE == :linear ? [:linear] :
                                      [:log]

    println("  Generando panel $(PANEL_ROWS)×$(PANEL_COLS)...")

    for scale in scales
        clims, transform, hm_cmap, cb_label, cb_ticks, file_tag, human_tag =
            prepare_yz_scale(scale, snap_files, final_yz)

        fig = Figure(size = (430 * PANEL_COLS + 90, 320 * PANEL_ROWS + 40))

        for (k, tref) in enumerate(use_times)
            _, i = findmin(abs.(ts .- tref))
            snap = all_snaps[i][2]

            row = div(k - 1, PANEL_COLS) + 1
            col = mod(k - 1, PANEL_COLS) + 1

            ax = Axis(fig[row, col];
                xlabel = row == PANEL_ROWS ? "z (a.u.)" : "",
                ylabel = col == 1 ? "y (a.u.)" : "",
                title  = @sprintf("t = %.1f a.u.", snap.t),
                titlesize = FONT_SIZE - 2,
                aspect = DataAspect()
            )

            rho_plot = transform(snap.rho)

            heatmap!(ax, snap.z, snap.y, rho_plot';
                colormap   = hm_cmap,
                colorrange = clims
            )

            draw_markers!(ax, snap.R1, snap.R2, snap.Rcm)
            apply_movie_view!(ax)
        end

        # Barra de color local: sólo junto al último panel de la última fila
        Colorbar(fig[PANEL_ROWS, PANEL_COLS + 1];
            colormap   = hm_cmap,
            colorrange = clims,
            label      = cb_label,
            ticks      = cb_ticks
        )

        colsize!(fig.layout, PANEL_COLS + 1, Fixed(28))
        colgap!(fig.layout, 0)

        save_fig_base(joinpath(out_dir, "snapshot_panel_" * file_tag), fig)

        empty!(fig)
        GC.gc(true)
    end

    println(" ✓ Panel guardado")
end

# ════════════════════════════════════════════════════════════════
#  PANEL: REGIÓN FIJA ALREDEDOR DEL ÁTOMO INICIAL
# ════════════════════════════════════════════════════════════════

function generate_initial_atom_region_panel(sim_dir::String, out_dir::String)
    snap_files = find_snapshots(sim_dir)
    isempty(snap_files) && return

    all_snaps = [(f, read_snapshot(f)) for f in snap_files]
    ts = [s[2].t for s in all_snaps]

    use_times = T_MAX_MOVIE > 0 ?
        filter(t -> t <= T_MAX_MOVIE, SNAP_TIMES) :
        copy(SNAP_TIMES)

    np = min(PANEL_ROWS * PANEL_COLS, length(use_times))
    np < 1 && return
    use_times = use_times[1:np]

    final_yz = joinpath(sim_dir, "dens_final_yz.dat")
    clims = estimate_log_clims(snap_files; extra_files = [final_yz])
    tvals, tlabels = log10_ticks(clims[1], clims[2])

    snap0 = all_snaps[1][2]
    zc = snap0.R1[3]
    yc = snap0.R1[2]

    zmin = zc - INITIAL_REGION_HALF_Z
    zmax = zc + INITIAL_REGION_HALF_Z
    ymin = yc - INITIAL_REGION_HALF_Y
    ymax = yc + INITIAL_REGION_HALF_Y

    println("  Generando región centrada en el átomo inicial...")

    fig = Figure(size = (400 * PANEL_COLS, 350 * PANEL_ROWS + 80))

    for (k, tref) in enumerate(use_times)
        _, i = findmin(abs.(ts .- tref))
        snap = all_snaps[i][2]

        row = div(k - 1, PANEL_COLS) + 1
        col = mod(k - 1, PANEL_COLS) + 1

        ax = Axis(fig[row, col];
            xlabel = row == PANEL_ROWS ? "z (a.u.)" : "",
            ylabel = col == 1 ? "y (a.u.)" : "",
            title  = @sprintf("t = %.1f a.u.", snap.t),
            titlesize = FONT_SIZE - 2,
            aspect = DataAspect()
        )

        rho_log = snapshot_to_logrho(snap.rho, clims)

        heatmap!(ax, snap.z, snap.y, rho_log';
            colormap   = MOVIE_COLORMAP,
            colorrange = clims
        )

        draw_markers!(ax, snap.R1, snap.R2, snap.Rcm)

        xlims!(ax, zmin, zmax)
        ylims!(ax, ymin, ymax)

        if INITIAL_REGION_SHOW_GUIDES
            vlines!(ax, [zc];
                color = INITIAL_REGION_GUIDE_COLOR,
                linewidth = 1.0,
                linestyle = :dash
            )

            hlines!(ax, [yc];
                color = INITIAL_REGION_GUIDE_COLOR,
                linewidth = 1.0,
                linestyle = :dash
            )
        end
    end

    Colorbar(fig[:, PANEL_COLS + 1];
        colormap   = MOVIE_COLORMAP,
        colorrange = clims,
        label      = rich("log", subscript("10"), " ρ(y,z,t)"),
        ticks      = (tvals, tlabels)
    )

    Label(fig[PANEL_ROWS + 1, 1:PANEL_COLS],
        @sprintf("Ventana fija centrada en R₁(t₀) = (z₀, y₀) = (%.3f, %.3f) a.u.", zc, yc);
        fontsize = FONT_SIZE - 2,
        tellwidth = false
    )

    save_fig_base(joinpath(out_dir, "initial_atom_region_panel"), fig)

    empty!(fig)
    GC.gc(true)
    println(" ✓ Región centrada en el átomo inicial guardada")
end

# ════════════════════════════════════════════════════════════════
#  PANEL: ZOOM NUCLEAR LOCAL
# ════════════════════════════════════════════════════════════════

function parse_symbol_string(s::AbstractString, default::Symbol)
    ss = lowercase(strip(s))
    ss == "x" && return :x
    ss == "y" && return :y
    ss == "z" && return :z
    return default
end

function infer_dirs_from_info(info::Dict{String,String})
    dir_collision = parse_symbol_string(info_get(info, "DIR_COLLISION", "z"), :z)
    dir_b         = parse_symbol_string(info_get(info, "DIR_B", "y"), :y)
    return dir_collision, dir_b
end

function yz_from_dirs(v::AbstractVector{<:Real}, dir_collision::Symbol, dir_b::Symbol)
    # Devuelve (coord_colisión, coord_impacto) proyectadas al plano visual yz.
    # En la convención usual de este postproceso:
    #   eje horizontal = coordenada de colisión
    #   eje vertical   = coordenada de impacto
    return (v[axis_index(dir_collision)], v[axis_index(dir_b)])
end

function axis_index(ax::Symbol)
    ax == :x && return 1
    ax == :y && return 2
    return 3
end

function nuclear_local_window(Tcoll, Tb, coll0, b0)
    # Devuelve ((H_lo, H_hi), (V_lo, V_hi)) según el modo configurado.
    # H = eje horizontal (colisión), V = eje vertical (impacto).
    if NUCLEAR_LOCAL_VIEW_MODE == :manual
        return NUCLEAR_LOCAL_H_AXIS, NUCLEAR_LOCAL_V_AXIS
    end
    # :auto — ventana centrada en el blanco inicial con padding 10 %
    # y semiancho mínimo de 1.0 a.u. para evitar ventanas degeneradas.
    hH = max(maximum(Tcoll) - coll0, coll0 - minimum(Tcoll), 1.0) * 1.10
    hV = max(maximum(Tb)    - b0,    b0    - minimum(Tb),    1.0) * 1.10
    return ((coll0 - hH, coll0 + hH), (b0 - hV, b0 + hV))
end

function generate_nuclear_local_zoom(sim_dir::String, out_dir::String)
    traj_file = joinpath(sim_dir, "traj_log.dat")
    isfile(traj_file) || return

    tr   = read_traj_log(traj_file)
    info = read_run_info(joinpath(sim_dir, "run_info.txt"))
    idx  = time_mask(tr.t, T_MAX_MOVIE)
    isempty(idx) && return

    println("  Generando zoom nuclear local en el plano de colisión...")

    # Ejes físicos del plano visual (no asume z–y)
    dir_collision, dir_b = infer_dirs_from_info(info)
    coll_str = axis_label_str(dir_collision)
    b_str    = axis_label_str(dir_b)

    # Coordenadas nucleares proyectadas al plano de colisión
    Tcoll = traj_coord(tr, :target, dir_collision)[idx]
    Tb    = traj_coord(tr, :target, dir_b)[idx]
    Pcoll = traj_coord(tr, :proj,   dir_collision)[idx]
    Pb    = traj_coord(tr, :proj,   dir_b)[idx]

    # Centro = posición inicial del blanco
    coll0 = Tcoll[1]
    b0    = Tb[1]

    # Ventana visual
    H_lim, V_lim = nuclear_local_window(Tcoll, Tb, coll0, b0)

    fig = Figure(size = (620, 620)) #puedes cambiar a (720, 720)

    ax = Axis(fig[1, 1];
        xlabel = "$coll_str (a.u.)",
        ylabel = "$b_str (a.u.)",
        title  = "Zoom local alrededor del blanco inicial",
        aspect = DataAspect()
    )

    lines!(ax, Tcoll, Tb;
        color = NUCLEAR_LOCAL_TARGET_COLOR,
        linewidth = NUCLEAR_LOCAL_TARGET_LW,
        label = "Blanco")

    lines!(ax, Pcoll, Pb;
        color = NUCLEAR_LOCAL_PROJ_COLOR,
        linewidth = NUCLEAR_LOCAL_PROJ_LW,
        label = "Proyectil")

    # Posiciones iniciales: círculo sólido
    scatter!(ax, [Tcoll[1]], [Tb[1]];
        color = NUCLEAR_LOCAL_TARGET_COLOR,
        marker = :circle, markersize = 10,
        strokecolor = :black, strokewidth = 1.2)

    scatter!(ax, [Pcoll[1]], [Pb[1]];
        color = NUCLEAR_LOCAL_PROJ_COLOR,
        marker = :circle, markersize = 10,
        strokecolor = :black, strokewidth = 1.2)

    # Posiciones finales: círculo vacío con strokecolor del átomo
    scatter!(ax, [Tcoll[end]], [Tb[end]];
        color = (:white, 0.0),
        marker = :circle, markersize = 10,
        strokecolor = NUCLEAR_LOCAL_TARGET_COLOR, strokewidth = 1.6)

    scatter!(ax, [Pcoll[end]], [Pb[end]];
        color = (:white, 0.0),
        marker = :circle, markersize = 10,
        strokecolor = NUCLEAR_LOCAL_PROJ_COLOR, strokewidth = 1.6)

    if NUCLEAR_LOCAL_SHOW_GUIDES
        vlines!(ax, [coll0]; color = NUCLEAR_LOCAL_GUIDE_COLOR,
                linestyle = :dash, linewidth = 1.0)
        hlines!(ax, [b0];    color = NUCLEAR_LOCAL_GUIDE_COLOR,
                linestyle = :dash, linewidth = 1.0)
    end

    xlims!(ax, H_lim...)
    ylims!(ax, V_lim...)

    axislegend(ax; position = :rb, labelsize = FONT_SIZE - 4)

    summary_txt =
        "Centro: ($(coll_str)₀, $(b_str)₀) = " *
        @sprintf("(%.3f, %.3f) a.u.\n", coll0, b0) *
        @sprintf("Ventana H (%s): [%.3f, %.3f] a.u.\n",
                 coll_str, H_lim[1], H_lim[2]) *
        @sprintf("Ventana V (%s): [%.3f, %.3f] a.u.   modo: %s",
                 b_str, V_lim[1], V_lim[2], String(NUCLEAR_LOCAL_VIEW_MODE))

    Label(fig[2, :], summary_txt;
        fontsize = FONT_SIZE - 2,
        tellwidth = false,
        tellheight = true)

    save_fig_base(joinpath(out_dir, "nuclear_local_zoom"), fig)

    empty!(fig)
    GC.gc(true)
    println("  ✓ Zoom nuclear local guardado")
end

# ════════════════════════════════════════════════════════════════
#  ANIMACIÓN DE TRAYECTORIAS NUCLEARES LOCALES
# ════════════════════════════════════════════════════════════════
function generate_movie_traj(sim_dir::String, out_dir::String)
    traj_file = joinpath(sim_dir, "traj_log.dat")
    isfile(traj_file) || return

    tr   = read_traj_log(traj_file)
    info = read_run_info(joinpath(sim_dir, "run_info.txt"))
    idx  = time_mask(tr.t, T_MAX_MOVIE)
    isempty(idx) && return

    # Ejes físicos del plano visual (no asume z–y)
    dir_collision, dir_b = infer_dirs_from_info(info)
    coll_str = axis_label_str(dir_collision)
    b_str    = axis_label_str(dir_b)

    # Metadatos para el título
    E_str = info_get(info, "E_keV", "?")
    b_par = info_get(info, "b",     "?")

    # Coordenadas crudas proyectadas al plano visual
    Tcoll = traj_coord(tr, :target, dir_collision)[idx]
    Tb    = traj_coord(tr, :target, dir_b)[idx]
    Pcoll = traj_coord(tr, :proj,   dir_collision)[idx]
    Pb    = traj_coord(tr, :proj,   dir_b)[idx]
    tt    = tr.t[idx]
    N     = length(tt)
    N == 0 && return

    println("  Generando animación de trayectorias nucleares ($N frames)...")

    # Centro = posición inicial del blanco; ventana según modo configurado
    coll0 = Tcoll[1]
    b0    = Tb[1]
    H_lim, V_lim = nuclear_local_window(Tcoll, Tb, coll0, b0)

    fig = Figure(size = (620, 620))

    subtitle_obs = Observable(@sprintf(
        "E = %s keV, b = %s, t = %.2f a.u.",
        E_str, b_par, tt[1]))

    ax = Axis(fig[1, 1];
        xlabel   = "$coll_str (a.u.)",
        ylabel   = "$b_str (a.u.)",
        title    = "Trayectorias nucleares locales",
        subtitle = subtitle_obs,
        aspect   = DataAspect()
    )

    xlims!(ax, H_lim...)
    ylims!(ax, V_lim...)

    # Trayectorias acumuladas (líneas sólidas, observables)
    Tcoll_obs = Observable([Tcoll[1]])
    Tb_obs    = Observable([Tb[1]])
    Pcoll_obs = Observable([Pcoll[1]])
    Pb_obs    = Observable([Pb[1]])

    # Posiciones actuales (marcadores pequeños, observables)
    Tcoll_now = Observable([Tcoll[1]])
    Tb_now    = Observable([Tb[1]])
    Pcoll_now = Observable([Pcoll[1]])
    Pb_now    = Observable([Pb[1]])

    # Líneas de trayectoria
    lines!(ax, Tcoll_obs, Tb_obs;
        color = NUCLEAR_LOCAL_TARGET_COLOR,
        linewidth = NUCLEAR_LOCAL_TARGET_LW,
        label = "Blanco")
    lines!(ax, Pcoll_obs, Pb_obs;
        color = NUCLEAR_LOCAL_PROJ_COLOR,
        linewidth = NUCLEAR_LOCAL_PROJ_LW,
        label = "Proyectil")

    # Posiciones iniciales: círculos sólidos fijos
    scatter!(ax, [Tcoll[1]], [Tb[1]];
        color = NUCLEAR_LOCAL_TARGET_COLOR,
        marker = :circle, markersize = 8,
        strokecolor = :black, strokewidth = 0.8)
    scatter!(ax, [Pcoll[1]], [Pb[1]];
        color = NUCLEAR_LOCAL_PROJ_COLOR,
        marker = :circle, markersize = 8,
        strokecolor = :black, strokewidth = 0.8)

    # Posiciones finales: círculos vacíos fijos
    scatter!(ax, [Tcoll[end]], [Tb[end]];
        color = (:white, 0.0),
        marker = :circle, markersize = 8,
        strokecolor = NUCLEAR_LOCAL_TARGET_COLOR, strokewidth = 1.6)
    scatter!(ax, [Pcoll[end]], [Pb[end]];
        color = (:white, 0.0),
        marker = :circle, markersize = 8,
        strokecolor = NUCLEAR_LOCAL_PROJ_COLOR, strokewidth = 1.6)

    # Marcadores de posición actual (pequeños, en movimiento)
    scatter!(ax, Tcoll_now, Tb_now;
        color = NUCLEAR_LOCAL_TARGET_COLOR,
        marker = :circle, markersize = NUCLEAR_LOCAL_NOW_MARKERSIZE)
    scatter!(ax, Pcoll_now, Pb_now;
        color = NUCLEAR_LOCAL_PROJ_COLOR,
        marker = :circle, markersize = NUCLEAR_LOCAL_NOW_MARKERSIZE)

    axislegend(ax; position = :rb, labelsize = FONT_SIZE - 4)

    movie_path = joinpath(out_dir, "nuclear_local_traj.mp4")

    record(fig, movie_path, 1:N; framerate = MOVIE_FPS) do i
        # Actualiza segmentos acumulados de trayectoria
        Tcoll_obs[] = Tcoll[1:i]
        Tb_obs[]    = Tb[1:i]
        Pcoll_obs[] = Pcoll[1:i]
        Pb_obs[]    = Pb[1:i]
        # Posiciones actuales
        Tcoll_now[] = [Tcoll[i]]
        Tb_now[]    = [Tb[i]]
        Pcoll_now[] = [Pcoll[i]]
        Pb_now[]    = [Pb[i]]
        # Título con tiempo actual
        subtitle_obs[] = @sprintf(
            "E = %s keV, b = %s, t = %.2f a.u.",
            E_str, b_par, tt[i])
    end

    empty!(fig)
    GC.gc(true)
    println("  ✓ Animación de trayectorias: $movie_path")
end
# ════════════════════════════════════════════════════════════════
#  TRAYECTORIAS
# ════════════════════════════════════════════════════════════════
function axis_symbol(s::String)
    ls = lowercase(strip(s))
    ls == "x" && return :x
    ls == "y" && return :y
    return :z
end

function axis_index(ax::Symbol)
    ax == :x && return 1
    ax == :y && return 2
    return 3
end

function trajectory_aspect(dir_b::Symbol, dir_collision::Symbol)
    asp = [1.0, 1.0, 1.0]
    asp[axis_index(dir_b)] = TRAJ_IMPACT_ASPECT
    asp[axis_index(dir_collision)] = TRAJ_COLLISION_ASPECT
    return Tuple(asp)
end

function set_centered_limits!(ax, tr, idx, dir_b::Symbol, dir_collision::Symbol)
    X = vcat(tr.R1x[idx], tr.R2x[idx])
    Y = vcat(tr.R1y[idx], tr.R2y[idx])
    Z = vcat(tr.R1z[idx], tr.R2z[idx])

    mins = [minimum(X), minimum(Y), minimum(Z)]
    maxs = [maximum(X), maximum(Y), maximum(Z)]
    ctr  = (mins .+ maxs) ./ 2
    span = maxs .- mins

    for i in 1:3
        span[i] = max(span[i], TRAJ_MIN_TRANSVERSE_SPAN)
    end
    span[axis_index(dir_collision)] = max(span[axis_index(dir_collision)], TRAJ_MIN_COLLISION_SPAN)

    half = span ./ 2
    pads = fill(TRAJ_TRANSVERSE_PAD, 3)
    pads[axis_index(dir_collision)] = TRAJ_COLLISION_PAD
    half .*= (1 .+ pads)

    xlims!(ax, ctr[1] - half[1], ctr[1] + half[1])
    ylims!(ax, ctr[2] - half[2], ctr[2] + half[2])
    zlims!(ax, ctr[3] - half[3], ctr[3] + half[3])
end


function generate_trajectory(sim_dir::String, out_dir::String)
    tfile = joinpath(sim_dir, "traj_log.dat")
    isfile(tfile) || return

    traj = read_traj_log(tfile)
    info = read_run_info(joinpath(sim_dir, "run_info.txt"))
    idx = time_mask(traj.t, T_MAX_MOVIE)
    isempty(idx) && return

    println("  Generando trayectorias 3D...")

    dir_collision = axis_symbol(info_get(info, "dir_collision", "z"))
    dir_b         = axis_symbol(info_get(info, "dir_b", "y"))

    fig = Figure(size=(950, 780))
    ax = Axis3(fig[1, 1];
        xlabel = "x (a.u.)",
        ylabel = "y (a.u.)",
        zlabel = "z (a.u.)",
        title  = "Trayectorias nucleares",
        perspectiveness = 0.35,
        aspect = trajectory_aspect(dir_b, dir_collision))

    lines!(ax, traj.R1x[idx], traj.R1y[idx], traj.R1z[idx];
        color = :blue, linewidth = 2.2, label = "Blanco")

    lines!(ax, traj.R2x[idx], traj.R2y[idx], traj.R2z[idx];
        color = :orange, linewidth = 2.2, label = "Proyectil")

    v0_str = info_get(info, "v0", "")
    v0 = tryparse(Float64, v0_str)

    if v0 !== nothing && length(idx) > 1
        tf_ = traj.t[idx[end]]
        rng = range(0.0, tf_; length=60)
        x0 = traj.R2x[1]
        y0 = traj.R2y[1]
        z0 = traj.R2z[1]

        if dir_collision == :x
            lines!(ax, [x0 + v0 * tt for tt in rng], fill(y0, length(rng)), fill(z0, length(rng));
                color = (:gray, 0.7), linewidth = 1.5, linestyle = :dash, label = "Rectilínea")
        elseif dir_collision == :y
            lines!(ax, fill(x0, length(rng)), [y0 + v0 * tt for tt in rng], fill(z0, length(rng));
                color = (:gray, 0.7), linewidth = 1.5, linestyle = :dash, label = "Rectilínea")
        else
            lines!(ax, fill(x0, length(rng)), fill(y0, length(rng)), [z0 + v0 * tt for tt in rng];
                color = (:gray, 0.7), linewidth = 1.5, linestyle = :dash, label = "Rectilínea")
        end
    end

    ie = idx[end]

    scatter!(ax, [traj.R1x[1]], [traj.R1y[1]], [traj.R1z[1]];
        marker = :circle, markersize = 10, color = :blue,
        strokewidth = 1.5, strokecolor = :black)

    scatter!(ax, [traj.R2x[1]], [traj.R2y[1]], [traj.R2z[1]];
        marker = :circle, markersize = 10, color = :orange,
        strokewidth = 1.5, strokecolor = :black)

    scatter!(ax, [traj.R1x[ie]], [traj.R1y[ie]], [traj.R1z[ie]];
        marker = :circle, markersize = 12, color = :blue)

    scatter!(ax, [traj.R2x[ie]], [traj.R2y[ie]], [traj.R2z[ie]];
        marker = :circle, markersize = 12, color = :orange)

    set_centered_limits!(ax, traj, idx, dir_b, dir_collision)
    axislegend(ax; position = :rt, labelsize = FONT_SIZE - 4)

    θ_str  = info_get(info, "theta_deg", "NaN")
    Pi_str = info_get(info, "Pi_mag", "")
    Pf_str = info_get(info, "Pf_mag", "")
    b_str  = info_get(info, "b", "")
    t_str  = info_get(info, "t_final", "")

    info_text = "θ = $(θ_str)°"
    !isempty(b_str)  && (info_text *= "\nb = $(b_str) a.u.")
    !isempty(Pi_str) && (info_text *= "\n|Pᵢ| = $(Pi_str)")
    !isempty(Pf_str) && (info_text *= "\n|Pf| = $(Pf_str)")
    !isempty(t_str)  && (info_text *= "\ntf = $(t_str) a.u.")

    Label(fig[2, 1], info_text;
        fontsize = FONT_SIZE - TRAJ_INFO_FONT_REDUCTION,
        halign = :right,
        valign = :top,
        tellwidth = false,
        tellheight = true)

    save_fig_base(joinpath(out_dir, "trajectories_3D"), fig)
    empty!(fig)
    GC.gc(true)
    println(" ✓ Trayectorias guardadas")
end


# ════════════════════════════════════════════════════════════════
#  DIAGNÓSTICOS DE ENERGÍA
# ════════════════════════════════════════════════════════════════

function generate_energy_plots(sim_dir::String, out_dir::String)
    efile = joinpath(sim_dir, "energy_log.dat")
    isfile(efile) || return

    E = read_energy_log(efile)
    idx = time_mask(E.t, T_MAX_MOVIE)
    isempty(idx) && return

    println("  Generando diagnósticos de energía...")

    fig = Figure(size=(850, 920))

    ax1 = Axis(fig[1, 1];
        ylabel = "Norma",
        title = "Norma")
    lines!(ax1, E.t[idx], E.norm[idx]; color=:black, linewidth=1.5)

    ax2 = Axis(fig[2, 1];
        ylabel = rich("E", subscript("elec"), " / ‖ψ‖²  (a.u.)"),
        title  = rich("Diagnóstico electrónico normalizado"))

    lines!(ax2, E.t[idx], E.T_e_n[idx];
        color = :purple, linewidth = 1.5,
        label = rich("T", subscript("e"), " / ‖ψ‖²"))

    lines!(ax2, E.t[idx], E.V_eN_n[idx];
        color = :red, linewidth = 1.5,
        label = rich("V", subscript("eN"), " / ‖ψ‖²"))

    lines!(ax2, E.t[idx], E.E_elec_n[idx];
        color = :goldenrod, linewidth = 1.7,
        label = rich("E", subscript("elec"), " / ‖ψ‖²"))

    axislegend(ax2; position=:rt, labelsize=FONT_SIZE - 4)

    ax3 = Axis(fig[3, 1];
        xlabel = "t (a.u.)",
        ylabel = "Energía (a.u.)",
        title  = rich("Componentes nucleares y energía total"))

    lines!(ax3, E.t[idx], E.K_P[idx];
        color = :purple, linewidth = 1.5,
        label = rich("K", subscript("P")))

    lines!(ax3, E.t[idx], E.K_T[idx];
        color = :cyan, linewidth = 1.5,
        label = rich("K", subscript("T")))

    lines!(ax3, E.t[idx], E.V_nn[idx];
        color = :red, linewidth = 1.5,
        label = rich("V", subscript("nn")))

    lines!(ax3, E.t[idx], E.E_total[idx];
        color = (:gray, 0.65), linewidth = 1.5,
        label = rich("E", subscript("total"), " raw"))

    lines!(ax3, E.t[idx], E.E_total_n[idx];
        color = :black, linewidth = 2.0,
        label = rich("E", subscript("total"), " norm"))

    axislegend(ax3; position=:rb, labelsize=FONT_SIZE - 4)

    save_fig_base(joinpath(out_dir, "energy_diagnostics"), fig)
    empty!(fig)
    GC.gc(true)
    println(" ✓ Energía guardada")
end

# ════════════════════════════════════════════════════════════════
#  DENSIDADES FINALES
# ════════════════════════════════════════════════════════════════

function generate_density_final(sim_dir::String, out_dir::String)
    dzfile = joinpath(sim_dir, "dens_final_z.dat")
    if isfile(dzfile)
        dz = read_density_z(dzfile)
        info = read_run_info(joinpath(sim_dir, "run_info.txt"))

        fig = Figure(size = (720, 470))
        ax = Axis(fig[1, 1];
            xlabel = "z (a.u.)",
            ylabel = "ρ(z)")

        lines!(ax, dz.z, dz.rho; color = DENSITY_COLOR, linewidth = 2)

        if SHOW_OMEGA_LINE
            omega_val = tryparse(Float64, info_get(info, "Omega", ""))
            if omega_val !== nothing
                vlines!(ax, [omega_val]; color = OMEGA_LINE_COLOR, linewidth = 1.6)
                ymax = maximum(dz.rho)
                if isfinite(ymax) && ymax > 0
                    text!(ax,
                        omega_val + OMEGA_TEXT_DX,
                        ymax * OMEGA_TEXT_Y_FRAC;
                        text = "Ω",
                        fontsize = FONT_SIZE + 6,
                        color = OMEGA_LINE_COLOR)
                end
            end
        end

        save_fig_base(joinpath(out_dir, "dens_final_z"), fig)
        empty!(fig)
        GC.gc(true)
    end

    dyzfile = joinpath(sim_dir, "dens_final_yz.dat")
    if isfile(dyzfile)
        snap = read_snapshot(dyzfile)
        snap_files = find_snapshots(sim_dir)

        scales = MOVIE_SCALE == :both   ? [:log, :linear] :
                 MOVIE_SCALE == :linear ? [:linear] :
                                          [:log]

        for scale in scales
            clims, transform, hm_cmap, cb_label, cb_ticks, file_tag, human_tag =
                prepare_yz_scale(scale, snap_files, dyzfile)

            fig = plot_snapshot(
                snap,
                clims,
                transform,
                hm_cmap,
                cb_label,
                cb_ticks;
                title_str = @sprintf("ρ(y,z) final  (t = %.1f a.u.)", snap.t)
            )

            save_fig_base(joinpath(out_dir, "dens_final_yz_" * file_tag), fig)
            empty!(fig)
            GC.gc(true)
        end
    end

    println(" ✓ Densidades finales guardadas")
end

# ════════════════════════════════════════════════════════════════
#  MOMENTO FINAL kz
# ════════════════════════════════════════════════════════════════

function generate_momentum(sim_dir::String, out_dir::String)
    kfile = joinpath(sim_dir, "dens_final_kz.dat")
    isfile(kfile) || return

    kz = read_momentum_kz(kfile)
    fig = Figure(size = (640, 420))
    ax = Axis(fig[1, 1];
        xlabel = rich("k", subscript("z"), " (a.u.)"),
        ylabel = rich("ρ(k", subscript("z"), ")"),
        title  = "Distribución en momento")

    lines!(ax, kz.kz, kz.rho; color = :black, linewidth = 2)
    save_fig_base(joinpath(out_dir, "dens_final_kz"), fig)
    empty!(fig)
    GC.gc(true)
    println(" ✓ Momento guardado")
end

# ════════════════════════════════════════════════════════════════
#  PLANO DE COLISIÓN
# ════════════════════════════════════════════════════════════════
function generate_collision_plane_diagnostic(sim_dir::String, out_dir::String)
    traj_file = joinpath(sim_dir, "traj_log.dat")
    isfile(traj_file) || return

    tr = read_traj_log(traj_file)
    info = read_run_info(joinpath(sim_dir, "run_info.txt"))
    idx = time_mask(tr.t, T_MAX_MOVIE)
    isempty(idx) && return

    println("  Generando diagnóstico del plano de colisión...")

    # Dirección de colisión y eje del parámetro de impacto
    dir_collision = axis_symbol(info_get(info, "dir_collision", "z"))
    dir_b         = axis_symbol(info_get(info, "dir_b", "y"))

    # Eje perpendicular al plano de colisión esperado
    dir_out = remaining_axis(dir_collision, dir_b)

    t = tr.t[idx]

    qT_out = traj_coord(tr, :target, dir_out)[idx]
    qP_out = traj_coord(tr, :proj,   dir_out)[idx]
    qC_out = traj_coord(tr, :cm,     dir_out)[idx]

    qT_b = traj_coord(tr, :target, dir_b)[idx]
    qP_b = traj_coord(tr, :proj,   dir_b)[idx]

    qT_col = traj_coord(tr, :target, dir_collision)[idx]
    qP_col = traj_coord(tr, :proj,   dir_collision)[idx]

    # Derivas respecto al valor inicial
    dqT_out = qT_out .- qT_out[1]
    dqP_out = qP_out .- qP_out[1]
    dqC_out = qC_out .- qC_out[1]

    max_drift = maximum(abs.(vcat(dqT_out, dqP_out, dqC_out)))
    rms_drift = sqrt(mean(vcat(dqT_out, dqP_out, dqC_out).^2))

    plane_str = axis_label_str(dir_b) * axis_label_str(dir_collision)
    out_str   = axis_label_str(dir_out)

    fig = Figure(size = (980, 760))

    # ── Panel 1: coordenada fuera del plano ───────────────────
    ax1 = Axis(fig[1, 1];
        xlabel = "t (a.u.)",
        ylabel = "$(out_str) (a.u.)",
        title  = "Coordenada fuera del plano esperado ($plane_str)"
    )

    lines!(ax1, t, qT_out; color = CPLANE_COLOR_TARGET, linewidth = 2.0, label = "Blanco")
    lines!(ax1, t, qP_out; color = CPLANE_COLOR_PROJ,   linewidth = 2.0, label = "Proyectil")
    lines!(ax1, t, qC_out; color = CPLANE_COLOR_CM,     linewidth = 1.6, label = "CM")
    axislegend(ax1; position = :rb, labelsize = FONT_SIZE - 4)

    # ── Panel 2: deriva fuera del plano ───────────────────────
    ax2 = Axis(fig[2, 1];
        xlabel = "t (a.u.)",
        ylabel = "Δ$out_str (a.u.)",
        title  = "Deriva fuera del plano"
    )

    hlines!(ax2, [0.0]; color = CPLANE_ZERO_COLOR, linewidth = 1.0, linestyle = :dash)
    lines!(ax2, t, dqT_out; color = CPLANE_COLOR_TARGET, linewidth = 2.0, label = "Δ$out_str blanco")
    lines!(ax2, t, dqP_out; color = CPLANE_COLOR_PROJ,   linewidth = 2.0, label = "Δ$out_str proyectil")
    lines!(ax2, t, dqC_out; color = CPLANE_COLOR_CM,     linewidth = 1.6, label = "Δ$out_str CM")
    axislegend(ax2; position = :rb, labelsize = FONT_SIZE - 4)

    # ── Panel 3: eje del parámetro de impacto ─────────────────
    bstr = axis_label_str(dir_b)
    ax3 = Axis(fig[1, 2];
        xlabel = "t (a.u.)",
        ylabel = "$bstr (a.u.)",
        title  = "Componente sobre el eje de b"
    )

    lines!(ax3, t, qT_b; color = CPLANE_COLOR_TARGET, linewidth = 2.0, label = "Blanco")
    lines!(ax3, t, qP_b; color = CPLANE_COLOR_PROJ,   linewidth = 2.0, label = "Proyectil")
    axislegend(ax3; position = :rb, labelsize = FONT_SIZE - 4)

    # ── Panel 4: eje de colisión ──────────────────────────────
    cstr = axis_label_str(dir_collision)
    ax4 = Axis(fig[2, 2];
        xlabel = "t (a.u.)",
        ylabel = "$cstr (a.u.)",
        title  = "Componente sobre el eje de colisión"
    )

    lines!(ax4, t, qT_col; color = CPLANE_COLOR_TARGET, linewidth = 2.0, label = "Blanco")
    lines!(ax4, t, qP_col; color = CPLANE_COLOR_PROJ,   linewidth = 2.0, label = "Proyectil")
    axislegend(ax4; position = :rb, labelsize = FONT_SIZE - 4)

    # Resumen físico-numérico
    summary_txt =
        "Plano esperado: $plane_str\n" *
        "Coordenada fuera del plano: $out_str\n" *
        @sprintf("max|Δ%s| = %.3e a.u.\n", out_str, max_drift) *
        @sprintf("rms(Δ%s) = %.3e a.u.", out_str, rms_drift)

    Label(fig[3, :], summary_txt;
        fontsize = FONT_SIZE - 2,
        tellwidth = false,
        tellheight = true)

    save_fig_base(joinpath(out_dir, "collision_plane_diagnostic"), fig)

    empty!(fig)
    GC.gc(true)
    println(" ✓ Diagnóstico del plano guardado")
end


# ════════════════════════════════════════════════════════════════
#  MOMENTO ANGULAR NUCLEAR
# ════════════════════════════════════════════════════════════════

function generate_angular_momentum(sim_dir::String, out_dir::String)
    traj_file = joinpath(sim_dir, "traj_log.dat")
    isfile(traj_file) || return

    tr = read_traj_log(traj_file)
    info = read_run_info(joinpath(sim_dir, "run_info.txt"))
    idx = time_mask(tr.t, T_MAX_MOVIE)
    isempty(idx) && return

    println("  Generando diagnóstico de momento angular...")

    dir_collision, dir_b = infer_dirs_from_info(info)
    dir_normal = expected_normal_axis(dir_collision, dir_b)

    t = tr.t[idx]
    n = length(t)

    Lx = zeros(Float64, n)
    Ly = zeros(Float64, n)
    Lz = zeros(Float64, n)

    for (k, i) in enumerate(idx)
        r1 = [
            tr.R1x[i] - tr.CMx[i],
            tr.R1y[i] - tr.CMy[i],
            tr.R1z[i] - tr.CMz[i],
        ]
        r2 = [
            tr.R2x[i] - tr.CMx[i],
            tr.R2y[i] - tr.CMy[i],
            tr.R2z[i] - tr.CMz[i],
        ]
        p1 = [tr.P1x[i], tr.P1y[i], tr.P1z[i]]
        p2 = [tr.P2x[i], tr.P2y[i], tr.P2z[i]]

        L = cross(r1, p1) + cross(r2, p2)
        Lx[k] = L[1]
        Ly[k] = L[2]
        Lz[k] = L[3]
    end

    Lhx = similar(Lx)
    Lhy = similar(Ly)
    Lhz = similar(Lz)
    Lmag = similar(Lx)

    for k in eachindex(Lx)
        ux, uy, uz, nm = unit_or_nan(Lx[k], Ly[k], Lz[k])
        Lhx[k] = ux
        Lhy[k] = uy
        Lhz[k] = uz
        Lmag[k] = nm
    end

    k0 = findfirst(x -> isfinite(x) && x > 1e-30, Lmag)
    k0 === nothing && return
    L0 = [Lx[k0], Ly[k0], Lz[k0]]
    L0mag = Lmag[k0]

    Δθ = fill(NaN, n)
    for k in eachindex(Lx)
        if isfinite(Lmag[k]) && Lmag[k] > 1e-30
            cang = (Lx[k]*L0[1] + Ly[k]*L0[2] + Lz[k]*L0[3]) / (Lmag[k] * L0mag)
            cang = clamp(cang, -1.0, 1.0)
            Δθ[k] = acos(cang) * 180 / π
        end
    end

    ΔLmag = Lmag .- L0mag

    normal_txt = axis_label_str(dir_normal)
    plane_txt  = axis_label_str(dir_b) * axis_label_str(dir_collision)

    fig = Figure(size = (980, 800))

    ax1 = Axis(fig[1, 1];
        xlabel = "t (a.u.)",
        ylabel = "L (a.u.)",
        title  = "Componentes de L(t)"
    )
    lines!(ax1, t, Lx; color = :red,   linewidth = 2.0, label = "Lx")
    lines!(ax1, t, Ly; color = :green, linewidth = 2.0, label = "Ly")
    lines!(ax1, t, Lz; color = :blue,  linewidth = 2.0, label = "Lz")
    axislegend(ax1; position = :rb, labelsize = FONT_SIZE - 4)

    ax2 = Axis(fig[1, 2];
        xlabel = "t (a.u.)",
        ylabel = "Magnitud (a.u.)",
        title  = "|L(t)| y deriva de magnitud"
    )
    lines!(ax2, t, Lmag;  color = :black,  linewidth = 2.0, label = "|L|")
    lines!(ax2, t, ΔLmag; color = :purple, linewidth = 1.8, linestyle = :dash, label = "Δ|L|")
    axislegend(ax2; position = :rb, labelsize = FONT_SIZE - 4)

    ax3 = Axis(fig[2, 1];
        xlabel = "t (a.u.)",
        ylabel = "L̂",
        title  = "Dirección unitaria de L(t)"
    )
    lines!(ax3, t, Lhx; color = :red,   linewidth = 2.0, label = "L̂x")
    lines!(ax3, t, Lhy; color = :green, linewidth = 2.0, label = "L̂y")
    lines!(ax3, t, Lhz; color = :blue,  linewidth = 2.0, label = "L̂z")
    axislegend(ax3; position = :rb, labelsize = FONT_SIZE - 4)

    ax4 = Axis(fig[2, 2];
        xlabel = "t (a.u.)",
        ylabel = "Δθ(L,L₀) (deg)",
        title  = "Desviación angular respecto a L inicial"
    )
    lines!(ax4, t, Δθ; color = :black, linewidth = 2.0)

    max_dev = any(isfinite, Δθ) ? maximum(filter(isfinite, Δθ)) : NaN
    max_dL  = any(isfinite, ΔLmag) ? maximum(abs.(filter(isfinite, ΔLmag))) : NaN

    summary_txt =
        "Plano esperado: $plane_txt\n" *
        "Normal esperada: ±$normal_txt\n" *
        @sprintf("max Δθ = %.3e deg\n", max_dev) *
        @sprintf("max |Δ|L|| = %.3e a.u.", max_dL)

    Label(fig[3, :], summary_txt;
        fontsize = FONT_SIZE - 2,
        tellwidth = false,
        tellheight = true)

    save_fig_base(joinpath(out_dir, "angular_momentum_diagnostic"), fig)

    empty!(fig)
    GC.gc(true)
    println(" ✓ Momento angular guardado")
end

# ════════════════════════════════════════════════════════════════
#  DESCUBRIMIENTO DE SIMULACIONES
# ════════════════════════════════════════════════════════════════

function is_sim_dir(dir::String)
    isdir(dir) || return false
    has_run_info = isfile(joinpath(dir, "run_info.txt"))
    has_traj     = isfile(joinpath(dir, "traj_log.dat"))
    has_snaps    = isdir(joinpath(dir, "datos_dens_yz"))
    has_energy   = isfile(joinpath(dir, "energy_log.dat"))
    return has_run_info && (has_traj || has_snaps || has_energy)
end

function find_sim_dirs(root::String)
    sims = String[]
    isdir(root) || return sims

    if is_sim_dir(root)
        push!(sims, root)
        return sims
    end

    for (dp, dns, _) in walkdir(root)
        if is_sim_dir(dp)
            push!(sims, dp)
            empty!(dns)
        end
    end

    sort!(sims)
    return sims
end

# ════════════════════════════════════════════════════════════════
#  DRIVER DE UNA SIMULACIÓN
# ════════════════════════════════════════════════════════════════

function ensure_dir(path::String)
    isdir(path) || mkpath(path)
    return path
end

function process_sim(sim_dir::String)
    out_dir = ensure_dir(OUT_DIR)

    info = read_run_info(joinpath(sim_dir, "run_info.txt"))
    E_str = info_get(info, "E_keV", "?")
    b_str = info_get(info, "b", "?")

    println("────────────────────────────────────────────────────────────")
    println("Simulación:")
    println("  Carpeta : $sim_dir")
    println("  E_keV   : $E_str")
    println("  b       : $b_str")
    println("  Salida  : $out_dir")
    println("────────────────────────────────────────────────────────────")

    GEN_MOVIE               && generate_movie(sim_dir, out_dir)
    GEN_SNAPSHOTS           && generate_snapshots(sim_dir, out_dir)
    GEN_SNAPSHOT_PANEL      && generate_snapshot_panel(sim_dir, out_dir)
    GEN_INITIAL_ATOM_REGION && generate_initial_atom_region_panel(sim_dir, out_dir)
    GEN_NUCLEAR_LOCAL_ZOOM  && generate_nuclear_local_zoom(sim_dir, out_dir)
    GEN_MOVIE_TRAJ          && generate_movie_traj(sim_dir, out_dir)
    GEN_TRAJECTORY          && generate_trajectory(sim_dir, out_dir)
    GEN_ENERGY_PLOTS        && generate_energy_plots(sim_dir, out_dir)
    GEN_DENSITY_FINAL       && generate_density_final(sim_dir, out_dir)
    GEN_MOMENTUM            && generate_momentum(sim_dir, out_dir)
    GEN_COLLISION_PLANE     && generate_collision_plane_diagnostic(sim_dir, out_dir)
    GEN_ANGULAR_MOMENTUM    && generate_angular_momentum(sim_dir, out_dir)
end

# ════════════════════════════════════════════════════════════════
#  DRIVER PRINCIPAL
# ════════════════════════════════════════════════════════════════

function main()
    set_science_theme!()

    sims = find_sim_dirs(DATA_PATH)

    if isempty(sims)
        println("No se encontraron simulaciones válidas en:")
        println("  $DATA_PATH")
        return
    end

    if length(sims) == 1
        println("Simulación individual detectada")
    else
        println("Se detectaron $(length(sims)) simulaciones")
    end

    t0 = time()

    for sim_dir in sims
        try
            process_sim(sim_dir)
        catch err
            println("  ✗ Error procesando: $sim_dir")
            showerror(stdout, err)
            println()
        end
    end

    wall = time() - t0
    println("============================================================")
    println("$CODE_NAME v$CODE_VERSION")
    println("Procesamiento terminado en $(round(wall; digits = 2)) s")
    println("============================================================")
end

main()
