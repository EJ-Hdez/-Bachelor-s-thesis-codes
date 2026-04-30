# ════════════════════════════════════════════════════════════════
# ENDview.jl  v2.1.0
# Postprocesador gráfico para datos de LattEND.jl / END-Ionized.jl
#
# Recolecta recursivamente observables.dat de múltiples runsets,
# fusiona datos duplicados y genera figuras científicas para
# tesis y artículos.
#
# Funcionalidades:
#   ▸ Recolección recursiva de múltiples runsets
#   ▸ Fusión inteligente de duplicados (E, b)
#   ▸ Filtro por modo físico: captura / ionización / ambos
#   ▸ Curvas por energía: Pcap(b), Pion(b), bPcap(b), bPion(b)
#   ▸ Overlay multi-energía
#   ▸ Secciones eficaces σ(E)
#   ▸ Stopping cross sections S_e(E), S_n(E), S_total(E)
#   ▸ Mapas 2D (heatmap) y superficies 3D con piso + wireframe
#   ▸ Frames rotatorios + generación de GIF (ffmpeg concat)
#   ▸ Diagnósticos individuales: ρ(z), E(t), trayectorias
#   ▸ Energía normalizada (E_total_norm) cuando disponible
#   ▸ Omega dinámico leído de cada simulación
#   ▸ Control de memoria con GC entre figuras
#   ▸ Colormap por observable (cmap keyword)
#
# Historial:
#   v1.0–1.4 — Desarrollo como LattViz.jl
#   v2.0.0   — Renombrado a ENDview.jl, PHYSICS_MODE, GIF, GC
#   v2.1.0   — Energía normalizada (4 paneles), heatmap 2D,
#              superficies 3D con piso + wireframe, sin datos
#              de referencia (scripts de comparación separados)
# ════════════════════════════════════════════════════════════════

using CairoMakie
using Printf
using Dates
using Statistics

const VERSION = "2.1.0"

# ════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DEL USUARIO
#  Edita esta sección antes de cada ejecución.
# ════════════════════════════════════════════════════════════════

# ── Rutas ──────────────────────────────────────────────────────
# DATA_ROOT: carpeta raíz. Se buscarán todos los observables.dat
#   debajo de esta ruta, sin importar cuántos runsets haya.
# OUT_DIR: carpeta donde se crean figuras y tablas.
const DATA_ROOT = "/media/your/path/here"
const OUT_DIR   = ""/media/your/path/here/new/results"

# ── Filtros de datos ───────────────────────────────────────────
# DIR_FILTER: dirección de incidencia.
#   "z", "x", "y" → solo esa dirección
#   "all"          → mezcla todas
const DIR_FILTER = "z"

# E_SELECT: energías para curvas y plots individuales.
#   [] → procesa todas las encontradas.
const E_SELECT = Float64[]

# ── Manejo de duplicados ───────────────────────────────────────
# Si hay datos del mismo (E, b) en múltiples runsets:
const E_DIGITS = 3            # redondeo para agrupar energías
const B_DIGITS = 3            # redondeo para agrupar b
const DUPLICATE_RULE = :mean  # :mean, :last, :first

# ── Modo físico ────────────────────────────────────────────────
# :capture → Pcap, bPcap, σ_cap  (p+H, He²⁺+H)
# :ionize  → Pion, bPion, σ_ion  (p̄+H)
# :both    → todo (⚠ riesgo de saturar RAM!)
const PHYSICS_MODE = :ionize

# ── Interruptores ──────────────────────────────────────────────

# Curvas por energía y overlay
const GEN_CURVES     = true
const GEN_OVERLAY    = true
const OVERLAY_E_LIST = Float64[1.0, 5.0, 10.0, 20.0, 25.0, 50.0, 75.0, 100.0]     #collect(1.0:1.0:10.0)    #Float64[]  # vacío = todas

# Plots individuales por simulación (una figura por cada b)
# ⚠ Con muchos datos, activar estos puede saturar la RAM.
#   Usa E_SELECT para limitar, o déjalos en false.
const GEN_DENSITY    = true    # ρ(z)
const GEN_ENERGY     = true    # E_total(t), norm(t), componentes
const GEN_TRAJECTORY = true    # z(t), r₁₂(t), vista 2D

# Secciones eficaces y stopping (requiere ≥2 energíastrueconst GEN_SIGMA  = true
const GEN_SIGMA      = true
const GEN_STOPPING   = true

# Mapas 2D y 3D (requiere true energías con barridos de b)
const GEN_MAPS_2D    = true
const GEN_MAPS_3D    = true

# Frames rotatorios y GIF trueado
const GEN_3D_FRAMES  = true  # 72 capturas por observable
const GEN_3D_GIF     = true  # requiere ffmpeg instalado

# ── Formato de salida ──────────────────────────────────────────
const SAVE_PDF = true          # vectorial, para tesis
const SAVE_PNG = false          # raster, para presentaciones
const DPI      = 300

# ── Plano Ω ────────────────────────────────────────────────────
# Usa el valor de Ω de cada simulación (leído de observables.dat).
const SHOW_OMEGA = false

# ── Parámetros de mapas ────────────────────────────────────────
# https://docs.makie.org/stable/explanations/colors#Colormaps
const MAP_COLORMAP     = :bwr  #:bwr :seismic 
const MAP_LEVELS       = 30    # niveles de contorno en 2D (creo que esto no sirve para nada)
const MIN_ENERGIES_MAP = 2     # mínimo de energías para generar mapas
const MIN_B_MAP        = 2     # mínimo de b por energía

# ── Rotación 3D ────────────────────────────────────────────────
const FRAME_STEP_DEG = 5       # grados entre frames
const FRAME_ELEV     = 25      # elevación fija (grados) 

# ── Paleta de colores para overlay ─────────────────────────────
const PALETTE = [
    :blue, :red, :green, :orange, :purple,
    :brown, :cyan, :magenta, :olive, :teal,
    :navy, :coral, :gold, :indigo, :crimson,
]

# ════════════════════════════════════════════════════════════════
#  CONSTANTES FÍSICAS
# ════════════════════════════════════════════════════════════════

const A0_CM      = 0.529177210903e-8       # radio de Bohr en cm
const A0SQ_16    = (A0_CM^2) / 1e-16       # a₀² → 10⁻¹⁶ cm²
const A0SQ_15    = (A0_CM^2) / 1e-15       # a₀² → 10⁻¹⁵ cm²
const HARTREE_EV = 27.211386245988         # 1 Hartree en eV



# ════════════════════════════════════════════════════════════════
#  TEMA VISUAL (estilo tesis/artículo)
# ════════════════════════════════════════════════════════════════

function set_thesis_theme!()
    set_theme!(Theme(
        fontsize = 22,
        fonts = (; regular = "TeX Gyre Termes", bold = "TeX Gyre Termes Bold"),
        Axis = (
            xgridvisible  = false, ygridvisible  = false,
            xlabelsize    = 26,    ylabelsize    = 26,
            xticklabelsize = 18,   yticklabelsize = 18,
            titlesize     = 24,    spinewidth    = 1.6,
            xtickwidth    = 1.3,   ytickwidth    = 1.3,
            xminorticksvisible = true,
            yminorticksvisible = false,
            topspinevisible    = true,
            rightspinevisible  = true,
        ),
        Legend = (
            framevisible = false,
            labelsize    = 20,
            patchsize    = (30, 16),
        ),
    ))
end

# ════════════════════════════════════════════════════════════════
#  TIPO DE DATO CENTRAL
# ════════════════════════════════════════════════════════════════

struct SimRow
    dir::String
    EkeV::Float64
    b::Float64
    Pcap::Float64
    Ptar::Float64
    Pion::Float64
    norm::Float64
    bPcap::Float64
    bPion::Float64
    Omega::Float64
    t_final::Float64
    nsteps::Int
    sP_final::Float64
    sT_final::Float64
    r12_min::Float64
    dE_proj::Float64
    KT_recoil::Float64
    dE_total::Float64
    stop::String
    wall_sec::Float64
    sim_dir::String
end

# ════════════════════════════════════════════════════════════════
#  RECOLECTOR RECURSIVO
# ════════════════════════════════════════════════════════════════

"""Busca recursivamente todos los observables.dat bajo `root`."""
function find_obs_files(root::String)
    files = String[]
    for (dp, _, fns) in walkdir(root)
        for f in fns
            f == "observables.dat" && push!(files, joinpath(dp, f))
        end
    end
    return files
end

"""Lee un observables.dat. Infiere dir y E de la estructura de carpetas."""
function read_observables(path::String)
    rows = SimRow[]
    parent = dirname(path)
    edir = basename(parent)            # "25keV"
    ddir = basename(dirname(parent))   # "dir_z"

    # Dirección
    dir_str = startswith(ddir, "dir_") ? ddir[5:end] : "z"

    # Energía del nombre de carpeta
    EkeV = NaN
    try
        EkeV = parse(Float64, replace(replace(edir, "p" => "."), "keV" => ""))
    catch
    end

    open(path) do io
        for ln in eachline(io)
            s = strip(ln)
            (isempty(s) || startswith(s, "#")) && continue
            p = split(s)
            length(p) >= 18 || continue
            try
                b = parse(Float64, p[1])

                # Reconstruir ruta a la carpeta "b = X"
                bf = "b = " * replace(replace(@sprintf("%.6f", b), r"0+$" => ""), r"\.$" => "")
                sd = joinpath(parent, bf)
                !isdir(sd) && (sd = parent)

                push!(rows, SimRow(
                    dir_str, EkeV, b,
                    parse(Float64, p[2]),   # Pcap
                    parse(Float64, p[3]),   # Ptar
                    parse(Float64, p[4]),   # Pion
                    parse(Float64, p[5]),   # norm
                    parse(Float64, p[6]),   # bPcap
                    parse(Float64, p[7]),   # bPion
                    parse(Float64, p[8]),   # Omega
                    parse(Float64, p[9]),   # t_final
                    parse(Int,     p[10]),  # nsteps
                    parse(Float64, p[11]),  # sP_final
                    parse(Float64, p[12]),  # sT_final
                    parse(Float64, p[13]),  # r12_min
                    parse(Float64, p[14]),  # dE_proj
                    parse(Float64, p[15]),  # KT_recoil
                    parse(Float64, p[16]),  # dE_total
                    String(p[17]),          # stop
                    parse(Float64, p[18]),  # wall_sec
                    sd,
                ))
            catch
                continue
            end
        end
    end
    return rows
end

"""Recolecta todos los datos, filtra por dirección, fusiona duplicados."""
function collect_data(root::String)
    files = find_obs_files(root)
    println("  Encontrados $(length(files)) observables.dat")

    all = SimRow[]
    for f in files
        append!(all, read_observables(f))
    end

    # Filtro por dirección
    if DIR_FILTER != "all"
        all = filter(r -> r.dir == DIR_FILTER, all)
    end
    println("  Filas totales (dir=$DIR_FILTER): $(length(all))")

    # Agrupar por (E, b) redondeado
    Ek(E) = round(E; digits = E_DIGITS)
    Bk(b) = round(b; digits = B_DIGITS)

    grp = Dict{Tuple{Float64,Float64}, Vector{SimRow}}()
    for r in all
        push!(get!(grp, (Ek(r.EkeV), Bk(r.b)), SimRow[]), r)
    end

    # Resolver duplicados
    merged = SimRow[]
    for ((_, _), rs) in grp
        if length(rs) == 1
            push!(merged, rs[1])
        elseif DUPLICATE_RULE == :last
            push!(merged, rs[end])
        elseif DUPLICATE_RULE == :first
            push!(merged, rs[1])
        else  # :mean
            push!(merged, SimRow(
                rs[1].dir, rs[1].EkeV, rs[1].b,
                mean(r.Pcap for r in rs),
                mean(r.Ptar for r in rs),
                mean(r.Pion for r in rs),
                mean(r.norm for r in rs),
                mean(r.bPcap for r in rs),
                mean(r.bPion for r in rs),
                mean(r.Omega for r in rs),
                mean(r.t_final for r in rs),
                rs[end].nsteps,
                mean(r.sP_final for r in rs),
                mean(r.sT_final for r in rs),
                minimum(r.r12_min for r in rs),
                mean(r.dE_proj for r in rs),
                mean(r.KT_recoil for r in rs),
                mean(r.dE_total for r in rs),
                rs[end].stop,
                mean(r.wall_sec for r in rs),
                rs[end].sim_dir,
            ))
        end
    end

    sort!(merged; by = r -> (r.EkeV, r.b))
    energies = sort(unique(round.([r.EkeV for r in merged]; digits = E_DIGITS)))
    println("  Energías: $energies keV")
    println("  Filas fusionadas: $(length(merged))")
    return merged, energies
end

# ════════════════════════════════════════════════════════════════
#  LECTORES DE ARCHIVOS INDIVIDUALES
# ════════════════════════════════════════════════════════════════

"""Lee archivo de dos columnas numéricas."""
function read2col(path::String)
    x = Float64[]; y = Float64[]
    isfile(path) || return x, y
    open(path) do io
        for ln in eachline(io)
            s = strip(ln)
            (isempty(s) || startswith(s, "#")) && continue
            p = split(s)
            length(p) >= 2 || continue
            try
                push!(x, parse(Float64, p[1]))
                push!(y, parse(Float64, p[2]))
            catch
            end
        end
    end
    return x, y
end

"""Lee dens_final_*.dat → (s, ρ)."""
function read_density(sim_dir::String)
    for f in ["dens_final_z.dat", "dens_final_x.dat", "dens_final_y.dat"]
        fp = joinpath(sim_dir, f)
        isfile(fp) && return read2col(fp)
    end
    return Float64[], Float64[]
end

"""Lee energy_log.dat → NamedTuple de vectores (9 cols raw + 4 norm si disponibles)."""
function read_elog(sim_dir::String)
    fp = joinpath(sim_dir, "energy_log.dat")
    isfile(fp) || return nothing

    t = Float64[]; KT = Float64[]; KP = Float64[]; Vn = Float64[]
    Te = Float64[]; Ve = Float64[]; Ee = Float64[]; Et = Float64[]; nr = Float64[]
    # Columnas normalizadas (pueden no existir en datos antiguos)
    Te_n = Float64[]; Ve_n = Float64[]; Ee_n = Float64[]; Et_n = Float64[]
    has_norm = false

    open(fp) do io
        for ln in eachline(io)
            s = strip(ln)
            (isempty(s) || startswith(s, "#")) && continue
            p = split(s)
            length(p) >= 9 || continue
            try
                push!(t,  parse(Float64, p[1]))
                push!(KT, parse(Float64, p[2]))
                push!(KP, parse(Float64, p[3]))
                push!(Vn, parse(Float64, p[4]))
                push!(Te, parse(Float64, p[5]))
                push!(Ve, parse(Float64, p[6]))
                push!(Ee, parse(Float64, p[7]))
                push!(Et, parse(Float64, p[8]))
                push!(nr, parse(Float64, p[9]))
                if length(p) >= 13
                    push!(Te_n, parse(Float64, p[10]))
                    push!(Ve_n, parse(Float64, p[11]))
                    push!(Ee_n, parse(Float64, p[12]))
                    push!(Et_n, parse(Float64, p[13]))
                    has_norm = true
                end
            catch
            end
        end
    end

    if has_norm && length(Te_n) == length(t)
        return (t=t, K_T=KT, K_P=KP, V_nn=Vn, T_e=Te, V_eN=Ve,
                E_elec=Ee, E_total=Et, norm=nr,
                T_e_norm=Te_n, V_eN_norm=Ve_n, E_elec_norm=Ee_n, E_total_norm=Et_n,
                has_norm=true)
    else
        return (t=t, K_T=KT, K_P=KP, V_nn=Vn, T_e=Te, V_eN=Ve,
                E_elec=Ee, E_total=Et, norm=nr,
                T_e_norm=Float64[], V_eN_norm=Float64[], E_elec_norm=Float64[], E_total_norm=Float64[],
                has_norm=false)
    end
end

"""Lee traj_log.dat → NamedTuple con posiciones 3D."""
function read_traj(sim_dir::String)
    fp = joinpath(sim_dir, "traj_log.dat")
    isfile(fp) || return nothing

    t = Float64[]; xT = Float64[]; yT = Float64[]; zT = Float64[]
    xP = Float64[]; yP = Float64[]; zP = Float64[]

    open(fp) do io
        for ln in eachline(io)
            s = strip(ln)
            (isempty(s) || startswith(s, "#")) && continue
            p = split(s)
            length(p) >= 7 || continue
            try
                push!(t,  parse(Float64, p[1]))
                push!(xT, parse(Float64, p[2]))
                push!(yT, parse(Float64, p[3]))
                push!(zT, parse(Float64, p[4]))
                push!(xP, parse(Float64, p[5]))
                push!(yP, parse(Float64, p[6]))
                push!(zP, parse(Float64, p[7]))
            catch
            end
        end
    end
    return (t=t, xT=xT, yT=yT, zT=zT, xP=xP, yP=yP, zP=zP)
end

# ════════════════════════════════════════════════════════════════
#  UTILIDADES
# ════════════════════════════════════════════════════════════════

ensure_dir(p) = (isdir(p) || mkpath(p); p)

"""Integral trapezoidal."""
function trapz(x, y)
    n = length(x)
    n < 2 && return 0.0
    return sum(0.5 * (x[i+1] - x[i]) * (y[i+1] + y[i]) for i in 1:n-1)
end

gcol(i) = PALETTE[mod1(i, length(PALETTE))]

function Elab(E)
    E < 1   && return @sprintf("%.2f keV", E)
    E == floor(E) && return @sprintf("%.0f keV", E)
    return @sprintf("%.1f keV", E)
end

Etag(E) = replace(Elab(E), " " => "_")
btag(b) = replace(@sprintf("%.2f", b), "." => "p")

"""Filtra y ordena datos para una energía."""
function data_E(rows, E)
    sub = filter(r -> round(r.EkeV; digits=E_DIGITS) == round(E; digits=E_DIGITS), rows)
    sort!(sub; by = r -> r.b)
    return sub
end

"""¿Este modo de observable debe generarse dado PHYSICS_MODE?"""
want(mode::Symbol) = (PHYSICS_MODE == :both || mode == :both || mode == PHYSICS_MODE)

"""Guarda figura en PDF y/o PNG."""
function savefig(fig, dir, name)
    ensure_dir(dir)
    SAVE_PDF && save(joinpath(dir, name * ".pdf"), fig)
    SAVE_PNG && save(joinpath(dir, name * ".png"), fig, px_per_unit = DPI / 96)
    println("    → $name")
end

# ════════════════════════════════════════════════════════════════
#  CURVAS POR ENERGÍA (con filtro PHYSICS_MODE)
# ════════════════════════════════════════════════════════════════

function plot_curves(rows, E, out)
    sub = data_E(rows, E)
    length(sub) < 2 && return
    b  = [r.b for r in sub]
    el = Elab(E)
    ed = ensure_dir(joinpath(out, "curvas", Etag(E)))

    curves = [
        (r -> r.Pcap,  "Pcap",   "Pcap_vs_b",  :blue,      :capture),
        (r -> r.Pion,  "Pion",   "Pion_vs_b",   :red,       :ionize),
        (r -> r.bPcap, "b·Pcap", "bPcap_vs_b",  :blue,      :capture),
        (r -> r.bPion, "b·Pion", "bPion_vs_b",  :red,       :ionize),
        (r -> r.norm,  "‖ψ‖²",  "norm_vs_b",   :darkgreen, :both),
    ]

    for (yf, yl, yn, cl, mode) in curves
        want(mode) || continue

        y = [yf(r) for r in sub]
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1,1]; xlabel = "b (a.u.)", ylabel = yl, title = "$yl(b) — $el")
        lines!(ax, b, y; color = cl, linewidth = 2.2)
        scatter!(ax, b, y; color = cl, markersize = 6)

        # Anotación de σ
        if yn == "bPcap_vs_b"
            σ = 2π * trapz(b, y)
            text!(ax, 0.95, 0.95; text = @sprintf("σ_cap = %.3f a₀²", σ),
                  fontsize = 16, align = (:right, :top), space = :relative, color = :gray40)
        elseif yn == "bPion_vs_b"
            σ = 2π * trapz(b, y)
            text!(ax, 0.95, 0.95; text = @sprintf("σ_ion = %.3f a₀²", σ),
                  fontsize = 16, align = (:right, :top), space = :relative, color = :gray40)
        end

        savefig(fig, ed, yn)
    end
end

# ════════════════════════════════════════════════════════════════
#  OVERLAY MULTI-ENERGÍA
# ════════════════════════════════════════════════════════════════

function plot_overlay(rows, energies, out)
    elist = isempty(OVERLAY_E_LIST) ? energies :
            filter(E -> any(r -> round(r.EkeV; digits=E_DIGITS) == round(E; digits=E_DIGITS), rows),
                   OVERLAY_E_LIST)
    length(elist) < 1 && return
    od = ensure_dir(joinpath(out, "overlay"))

    overlays = [
        (r -> r.bPcap, "b·Pcap", "bPcap_overlay", :capture),
        (r -> r.bPion, "b·Pion", "bPion_overlay",  :ionize),
        (r -> r.Pcap,  "Pcap",   "Pcap_overlay",   :capture),
        (r -> r.Pion,  "Pion",   "Pion_overlay",   :ionize),
    ]

    for (yf, yl, yn, mode) in overlays
        want(mode) || continue

        fig = Figure(size = (900, 600))
        ax = Axis(fig[1,1]; xlabel = "b (a.u.)", ylabel = yl, title = "$yl(b) — multi-energía")
        for (i, E) in enumerate(elist)
            sub = data_E(rows, E)
            length(sub) < 2 && continue
            b = [r.b for r in sub]
            y = [yf(r) for r in sub]
            lines!(ax, b, y; color = gcol(i), linewidth = 1.8, label = Elab(E))
        end
        axislegend(ax, position = :rt)
        savefig(fig, od, yn)
    end
end

# ════════════════════════════════════════════════════════════════
#  PLOTS INDIVIDUALES (Omega dinámico)
# ════════════════════════════════════════════════════════════════

function plot_dens(row::SimRow, out)
    s, rho = read_density(row.sim_dir)
    isempty(s) && return
    el = Elab(row.EkeV)

    fig = Figure(size = (900, 550))
    ax = Axis(fig[1,1]; xlabel = "z (a.u.)", ylabel = "ρ(z)", title = "E = $el, b = $(row.b)")
    ax.yticks = 0.0:0.05:0.5
    lines!(ax, s, rho; color = :blue, linewidth = 2.2)

    if SHOW_OMEGA && isfinite(row.Omega)
        vlines!(ax, [row.Omega]; color = :purple, linestyle = :dot, linewidth = 2.0)
        text!(ax, row.Omega + 0.5, maximum(rho) * 0.9;
              text = "Ω", fontsize = 22, color = :purple)
    end

    savefig(fig, ensure_dir(joinpath(out, "density", Etag(row.EkeV))), "rho_b" * btag(row.b))
    empty!(fig)
end

function plot_ener(row::SimRow, out)
    d = read_elog(row.sim_dir)
    d === nothing && return
    isempty(d.t) && return
    el = Elab(row.EkeV)

    if d.has_norm
        # ── 4 paneles: E_total_norm, norma, desglose norm, componentes ──
        fig = Figure(size = (1100, 1100))

        # Panel 1: E_total_norm (principal) con E_total_raw de fondo
        ax1 = Axis(fig[1,1]; ylabel = "E_total (a.u.)",
                   title = "Energía — E = $el, b = $(row.b)")
        lines!(ax1, d.t, d.E_total;      color = :gray70, linewidth = 1.0, label = "raw")
        lines!(ax1, d.t, d.E_total_norm; color = :black,  linewidth = 1.8, label = "norm")
        axislegend(ax1, position = :lt, labelsize = 14)
        if length(d.E_total_norm) >= 2
            dE  = d.E_total_norm[end] - d.E_total_norm[1]
            rel = d.E_total_norm[1] != 0 ? abs(dE / d.E_total_norm[1]) : NaN
            text!(ax1, 0.02, 0.15; space = :relative, fontsize = 13, color = :gray40,
                  text = @sprintf("ΔE_norm = %.3e  ΔE/|E₀| = %.1e", dE, rel))
        end

        # Panel 2: Norma
        ax2 = Axis(fig[2,1]; ylabel = "‖ψ‖²")
        lines!(ax2, d.t, d.norm; color = :darkgreen, linewidth = 1.6)

        # Panel 3: Desglose electrónico normalizado
        ax3 = Axis(fig[3,1]; ylabel = "E_elec (norm, a.u.)")
        lines!(ax3, d.t, d.T_e_norm;    color = :blue,   linewidth = 1.3, label = "T_e/‖ψ‖²")
        lines!(ax3, d.t, d.V_eN_norm;   color = :red,    linewidth = 1.3, label = "V_eN/‖ψ‖²")
        lines!(ax3, d.t, d.E_elec_norm; color = :orange,  linewidth = 1.6, label = "E_elec/‖ψ‖²")
        axislegend(ax3, position = :rt, labelsize = 14)

        # Panel 4: Componentes nucleares + E_elec_norm
        ax4 = Axis(fig[4,1]; xlabel = "t (a.u.)", ylabel = "Energía (a.u.)")
        lines!(ax4, d.t, d.K_P;         color = :blue,   linewidth = 1.3, label = "K_P")
        lines!(ax4, d.t, d.K_T;         color = :cyan,   linewidth = 1.3, label = "K_T")
        lines!(ax4, d.t, d.V_nn;        color = :red,    linewidth = 1.3, label = "V_nn")
        lines!(ax4, d.t, d.E_elec_norm; color = :orange,  linewidth = 1.3, label = "E_elec_norm")
        axislegend(ax4, position = :rt, labelsize = 14)

    else
        # ── 3 paneles: formato original para datos sin normalizar ──
        fig = Figure(size = (1100, 850))

        ax1 = Axis(fig[1,1]; ylabel = "E_total (a.u.)",
                   title = "Energía — E = $el, b = $(row.b)")
        lines!(ax1, d.t, d.E_total; color = :black, linewidth = 1.6)
        if length(d.E_total) >= 2
            dE  = d.E_total[end] - d.E_total[1]
            rel = d.E_total[1] != 0 ? abs(dE / d.E_total[1]) : NaN
            text!(ax1, 0.02, 0.15; space = :relative, fontsize = 13, color = :gray40,
                  text = @sprintf("ΔE = %.3e  ΔE/|E₀| = %.1e", dE, rel))
        end

        ax2 = Axis(fig[2,1]; ylabel = "‖ψ‖²")
        lines!(ax2, d.t, d.norm; color = :darkgreen, linewidth = 1.6)

        ax3 = Axis(fig[3,1]; xlabel = "t (a.u.)", ylabel = "Energía (a.u.)")
        lines!(ax3, d.t, d.K_P;    color = :blue,   linewidth = 1.3, label = "K_P")
        lines!(ax3, d.t, d.K_T;    color = :cyan,   linewidth = 1.3, label = "K_T")
        lines!(ax3, d.t, d.V_nn;   color = :red,    linewidth = 1.3, label = "V_nn")
        lines!(ax3, d.t, d.E_elec; color = :orange,  linewidth = 1.3, label = "E_elec")
        axislegend(ax3, position = :rt)
    end

    savefig(fig, ensure_dir(joinpath(out, "energy", Etag(row.EkeV))), "energy_b" * btag(row.b))
    empty!(fig)
end

function plot_traj(row::SimRow, out)
    tj = read_traj(row.sim_dir)
    tj === nothing && return
    isempty(tj.t) && return
    el = Elab(row.EkeV)
    td = ensure_dir(joinpath(out, "trayectorias", Etag(row.EkeV)))

    # z(t) + r₁₂(t)
    fig = Figure(size = (1000, 700))
    ax1 = Axis(fig[1,1]; ylabel = "z (a.u.)", title = "Trayectoria — E = $el, b = $(row.b)")
    lines!(ax1, tj.t, tj.zT; color = :blue, linewidth = 1.8, label = "Target")
    lines!(ax1, tj.t, tj.zP; color = :red,  linewidth = 1.8, label = "Proyectil")
    if SHOW_OMEGA && isfinite(row.Omega)
        hlines!(ax1, [row.Omega]; color = :purple, linestyle = :dot, linewidth = 1.2)
    end
    axislegend(ax1, position = :lt)

    r12 = @. sqrt((tj.xP - tj.xT)^2 + (tj.yP - tj.yT)^2 + (tj.zP - tj.zT)^2)
    ax2 = Axis(fig[2,1]; xlabel = "t (a.u.)", ylabel = "r₁₂ (a.u.)")
    lines!(ax2, tj.t, r12; color = :black, linewidth = 1.8)
    rm = minimum(r12)
    tm = tj.t[argmin(r12)]
    scatter!(ax2, [tm], [rm]; color = :red, markersize = 8)
    text!(ax2, tm + 1.0, rm + 0.5; fontsize = 14, color = :red,
          text = @sprintf("r₁₂_min = %.3f", rm))

    savefig(fig, td, "traj_b" * btag(row.b))
    empty!(fig)

    # Vista 2D (z, y)
    fig2 = Figure(size = (900, 550))
    ax = Axis(fig2[1,1]; xlabel = "z (a.u.)", ylabel = "y (a.u.)",
              title = "Trayectoria (z, y) — E = $el, b = $(row.b)")
    lines!(ax, tj.zT, tj.yT; color = :blue, linewidth = 2.0, label = "Target")
    lines!(ax, tj.zP, tj.yP; color = :red,  linewidth = 2.0, label = "Proyectil")
    scatter!(ax, [tj.zP[1]],   [tj.yP[1]];   color = :red, marker = :circle, markersize = 10)
    scatter!(ax, [tj.zP[end]], [tj.yP[end]]; color = :red, marker = :star5,  markersize = 12)
    axislegend(ax, position = :lt)

    savefig(fig2, td, "traj2d_b" * btag(row.b))
    empty!(fig2)
end

# ════════════════════════════════════════════════════════════════
#  TABLAS DE RESUMEN
# ════════════════════════════════════════════════════════════════

function write_tables(rows, energies, out)
    ensure_dir(out)

    # Tabla fusionada
    open(joinpath(out, "merged_data.dat"), "w") do io
        println(io, "# ENDview.jl v$VERSION — Datos fusionados")
        println(io, "# E(keV)  b  Pcap  Ptar  Pion  norm  bPcap  bPion  ΔE_proj  KT_recoil  stop")
        for r in rows
            @printf(io, "%10.4f  %10.6f  %12.6e  %12.6e  %12.6e  %12.8f  %12.6e  %12.6e  %16.8e  %16.8e  %s\n",
                    r.EkeV, r.b, r.Pcap, r.Ptar, r.Pion, r.norm,
                    r.bPcap, r.bPion, r.dE_proj, r.KT_recoil, r.stop)
        end
    end
    println("  → merged_data.dat ($(length(rows)) filas)")

    # σ(E)
    open(joinpath(out, "sigma_vs_E.dat"), "w") do io
        println(io, "# E(keV)  σ_cap(a₀²)  σ_cap(1e-16cm²)  σ_ion(a₀²)  σ_ion(1e-16cm²)")
        for E in energies
            sub = data_E(rows, E)
            length(sub) < 2 && continue
            b = [r.b for r in sub]
            σc = 2π * trapz(b, [r.bPcap for r in sub])
            σi = 2π * trapz(b, [r.bPion for r in sub])
            @printf(io, "%10.4f  %12.6f  %12.6f  %12.6f  %12.6f\n",
                    E, σc, σc * A0SQ_16, σi, σi * A0SQ_16)
        end
    end
    println("  → sigma_vs_E.dat")
end

# ════════════════════════════════════════════════════════════════
#  SECCIONES EFICACES Y STOPPING
# ════════════════════════════════════════════════════════════════

"""Calcula σ_cap, σ_ion, S_elec, S_nuc para cada energía."""
function compute_cross_sections(rows, energies)
    E_out = Float64[]; σ_cap = Float64[]; σ_ion = Float64[]
    S_elec = Float64[]; S_nuc = Float64[]

    for E in energies
        sub = data_E(rows, E)
        length(sub) < 2 && continue
        b = [r.b for r in sub]

        push!(E_out, E)
        push!(σ_cap,  2π * trapz(b, [r.bPcap for r in sub]))
        push!(σ_ion,  2π * trapz(b, [r.bPion for r in sub]))
        push!(S_elec, 2π * trapz(b, [abs(r.dE_proj) * r.b for r in sub]))
        push!(S_nuc,  2π * trapz(b, [r.KT_recoil * r.b for r in sub]))
    end

    return (E = E_out, σ_cap = σ_cap, σ_ion = σ_ion, S_elec = S_elec, S_nuc = S_nuc)
end

"""σ_cap(E)."""
function plot_sigma_cap(cs, out)
    isempty(cs.E) && return
    sd = ensure_dir(joinpath(out, "sigma"))
    σ16 = cs.σ_cap .* A0SQ_16

    fig = Figure(size = (1000, 700))
    ax = Axis(fig[1,1]; xlabel = "E (keV)", ylabel = "σ_cap (10⁻¹⁶ cm²)",
              xscale = log10, title = "Sección eficaz de captura electrónica")
    lines!(ax, cs.E, σ16; color = :blue, linewidth = 2.8, label = "LattEND")
    scatter!(ax, cs.E, σ16; color = :blue, markersize = 8)
    axislegend(ax, position = :rt)
    savefig(fig, sd, "sigma_cap_vs_E")
end

"""σ_ion(E)."""
function plot_sigma_ion(cs, out)
    isempty(cs.E) && return
    sd = ensure_dir(joinpath(out, "sigma"))
    σ16 = cs.σ_ion .* A0SQ_16

    fig = Figure(size = (1000, 700))
    ax = Axis(fig[1,1]; xlabel = "E (keV)", ylabel = "σ_ion (10⁻¹⁶ cm²)",
              xscale = log10, title = "Sección eficaz de ionización")
    lines!(ax, cs.E, σ16; color = :red, linewidth = 2.8, label = "LattEND")
    scatter!(ax, cs.E, σ16; color = :red, markersize = 8)
    axislegend(ax, position = :rt)
    savefig(fig, sd, "sigma_ion_vs_E")
end

"""Stopping cross sections S_e(E), S_n(E), S_total(E)."""
function plot_stopping(cs, out)
    isempty(cs.E) && return
    sd = ensure_dir(joinpath(out, "sigma"))

    Se15 = cs.S_elec .* HARTREE_EV .* A0SQ_15
    Sn15 = cs.S_nuc  .* HARTREE_EV .* A0SQ_15
    St15 = Se15 .+ Sn15

    fig = Figure(size = (1000, 700))
    ax = Axis(fig[1,1]; xlabel = "E (keV)", ylabel = "S (10⁻¹⁵ eV·cm²)",
              xscale = log10, title = "Stopping cross sections")
    lines!(ax, cs.E, St15; color = :black,  linewidth = 2.5, label = "S_total")
    lines!(ax, cs.E, Se15; color = :green,  linewidth = 2.0, label = "S_elec")
    lines!(ax, cs.E, Sn15; color = :cyan,   linewidth = 2.0, label = "S_nuc")
    scatter!(ax, cs.E, St15; color = :black, markersize = 7)
    scatter!(ax, cs.E, Se15; color = :green, markersize = 6)
    scatter!(ax, cs.E, Sn15; color = :cyan,  markersize = 6)
    axislegend(ax, position = :rt)
    savefig(fig, sd, "stopping_vs_E")
end

"""Tabla de stopping."""
function write_stopping_table(cs, out)
    isempty(cs.E) && return
    sd = ensure_dir(joinpath(out, "sigma"))
    Se15 = cs.S_elec .* HARTREE_EV .* A0SQ_15
    Sn15 = cs.S_nuc  .* HARTREE_EV .* A0SQ_15

    open(joinpath(sd, "stopping_vs_E.dat"), "w") do io
        println(io, "# E(keV)  S_elec  S_nuc  S_total  (10⁻¹⁵ eV·cm²)")
        for i in eachindex(cs.E)
            @printf(io, "%10.4f  %14.6e  %14.6e  %14.6e\n",
                    cs.E[i], Se15[i], Sn15[i], Se15[i] + Sn15[i])
        end
    end
    println("  → stopping_vs_E.dat")
end

# ════════════════════════════════════════════════════════════════
#  MAPAS 2D, SUPERFICIES 3D, FRAMES ROTATORIOS, GIF
# ════════════════════════════════════════════════════════════════

"""Construye matrices Z(b, E) para todos los observables."""
function build_map_matrices(rows, energies)
    all_b = sort(unique(round.([r.b for r in rows]; digits = B_DIGITS)))
    nE = length(energies)
    nb = length(all_b)

    Pc  = fill(NaN, nb, nE);  Pi  = fill(NaN, nb, nE)
    bPc = fill(NaN, nb, nE);  bPi = fill(NaN, nb, nE)
    nm  = fill(NaN, nb, nE);  dEp = fill(NaN, nb, nE)

    for (jE, E) in enumerate(energies)
        for r in data_E(rows, E)
            ib = findfirst(x -> abs(x - r.b) < 10.0^(-B_DIGITS), all_b)
            ib === nothing && continue
            Pc[ib,jE]  = r.Pcap;  Pi[ib,jE]  = r.Pion
            bPc[ib,jE] = r.bPcap; bPi[ib,jE] = r.bPion
            nm[ib,jE]  = r.norm;  dEp[ib,jE] = abs(r.dE_proj)
        end
    end

    return (b = all_b, E = energies,
            Pcap = Pc, Pion = Pi, bPcap = bPc, bPion = bPi,
            norm = nm, dE_proj = dEp)
end

"""Extrae subconjunto válido de una matriz, descartando filas/cols con NaN."""
function valid_subset(mat, b_grid, E_grid)
    vj = [j for j in 1:length(E_grid) if count(!isnan, mat[:, j]) >= MIN_B_MAP]
    vi = [i for i in 1:length(b_grid) if count(!isnan, mat[i, :]) >= 1]
    (length(vj) < MIN_ENERGIES_MAP || length(vi) < MIN_B_MAP) && return nothing
    return (b    = b_grid[vi],
            E    = E_grid[vj],
            Z    = replace(mat[vi, vj], NaN => 0.0),
            logE = log10.(E_grid[vj]))
end

"""Heatmap 2D estilo pcolormesh (celdas pixeladas, no contornos)."""
function plot_map2d(mat, b_grid, E_grid, zlabel, title_str, out_dir, filename; cmap=MAP_COLORMAP)
    vs = valid_subset(mat, b_grid, E_grid)
    vs === nothing && return

    fig = Figure(size = (1000, 700))
    ax = Axis(fig[1,1];
        xlabel = "b (a.u.)",
        ylabel = "E (keV, escala log)",
        title  = title_str,
    )

    # heatmap espera Z[nx, ny] donde nx=length(x), ny=length(y)
    # nuestra Z tiene filas=b, cols=E → transponer para que x=b, y=logE
    hm = heatmap!(ax, vs.b, vs.logE, vs.Z;
        colormap = cmap,
    )
    Colorbar(fig[1,2], hm; label = zlabel)

    # Etiquetas del eje y como energía real en keV
    tl = [E < 1 ? @sprintf("%.2f", E) : (E == floor(E) ? @sprintf("%.0f", E) : @sprintf("%.1f", E)) for E in vs.E]
    if length(vs.logE) > 10
        st = max(1, length(vs.logE) ÷ 10)
        ix = 1:st:length(vs.logE)
        ax.yticks = (vs.logE[ix], tl[ix])
    else
        ax.yticks = (vs.logE, tl)
    end

    savefig(fig, ensure_dir(out_dir), filename)
    empty!(fig)
end

"""Superficie 3D con proyección al piso.
   Ejes: b horizontal (x), log₁₀E profundidad (y), Z vertical."""
function plot_surf3d(mat, b_grid, E_grid, zlabel, title_str, out_dir, filename; cmap=MAP_COLORMAP)
    vs = valid_subset(mat, b_grid, E_grid)
    vs === nothing && return

    zmin_data = minimum(vs.Z)
    zmax_data = maximum(vs.Z)
    z_floor = 0.0

    # Ticks de energía para el eje Y
    tl = [E < 1 ? @sprintf("%.2f", E) : (E == floor(E) ? @sprintf("%.0f", E) : @sprintf("%.1f", E)) for E in vs.E]
    if length(vs.logE) > 8
        st = max(1, length(vs.logE) ÷ 8)
        ix = collect(1:st:length(vs.logE))
    else
        ix = collect(1:length(vs.logE))
    end

    fig = Figure(size = (1100, 800))
    ax = Axis3(fig[1,1];
        xlabel    = "b (a.u.)",
        ylabel    = "E (keV)",
        zlabel    = zlabel,
        title     = title_str,
        azimuth   = -60.0 * π / 180,
        elevation = 25.0 * π / 180,
        yticks    = (vs.logE[ix], tl[ix]),
    )

    # Z tiene forma [nb, nE], surface!(ax, x, y, Z) con x=b, y=logE
    floor_Z = fill(Float32(z_floor), length(vs.b), length(vs.logE))

    # Piso coloreado
    surface!(ax, vs.b, vs.logE, floor_Z;
        color    = Float32.(vs.Z),
        colormap = cmap,
        shading  = NoShading,
        alpha    = 0.90,
    )

    # Superficie principal semi-transparente
    surface!(ax, vs.b, vs.logE, Float32.(vs.Z);
        colormap = cmap,
        shading  = NoShading,
        alpha    = 0.75,
    )

    # Curvas de nivel: una línea por cada energía
    for jE in 1:length(vs.logE)
        lines!(ax, vs.b, fill(vs.logE[jE], length(vs.b)), vs.Z[:, jE];
            color = :black, linewidth = 0.6, alpha = 0.5)
    end
    # Curvas por cada b (dirección transversal)
    for ib in 1:4:length(vs.b)   # cada 4 para no saturar
        lines!(ax, fill(vs.b[ib], length(vs.logE)), vs.logE, vs.Z[ib, :];
            color = :black, linewidth = 0.4, alpha = 0.4)
    end

    zlims!(ax, z_floor, 1.06 * zmax_data)
    Colorbar(fig[1,2]; colormap = cmap,
             limits = (zmin_data, zmax_data), label = zlabel)

    savefig(fig, ensure_dir(out_dir), filename)
    empty!(fig)
end

"""Frames rotatorios con piso + GIF."""
function plot_3d_frames(mat, b_grid, E_grid, zlabel, title_str, out_dir, tag; cmap=MAP_COLORMAP)
    vs = valid_subset(mat, b_grid, E_grid)
    vs === nothing && return

    fdir = ensure_dir(joinpath(out_dir, "frames_$tag"))
    elev_rad = FRAME_ELEV * π / 180
    zmin_data = minimum(vs.Z)
    zmax_data = maximum(vs.Z)
    z_floor = 0.0

    tl = [E < 1 ? @sprintf("%.2f", E) : (E == floor(E) ? @sprintf("%.0f", E) : @sprintf("%.1f", E)) for E in vs.E]
    if length(vs.logE) > 8
        st = max(1, length(vs.logE) ÷ 8)
        ix = collect(1:st:length(vs.logE))
    else
        ix = collect(1:length(vs.logE))
    end

    frame_paths = String[]
    floor_Z = fill(Float32(z_floor), length(vs.b), length(vs.logE))

    for az_deg in 0:FRAME_STEP_DEG:355
        fig = Figure(size = (1100, 800))
        ax = Axis3(fig[1,1];
            xlabel    = "b (a.u.)",
            ylabel    = "E (keV)",
            zlabel    = zlabel,
            title     = title_str,
            azimuth   = az_deg * π / 180,
            elevation = elev_rad,
            yticks    = (vs.logE[ix], tl[ix]),
        )

        surface!(ax, vs.b, vs.logE, floor_Z;
            color    = Float32.(vs.Z),
            colormap = cmap,
            shading  = NoShading,
            alpha    = 0.90,
        )

            # Superficie principal semi-transparente
        surface!(ax, vs.b, vs.logE, Float32.(vs.Z);
            colormap = cmap,
            shading  = NoShading,
            alpha    = 0.75,
        )
        
        # Curvas de nivel: una línea por cada energía
        for jE in 1:length(vs.logE)
            lines!(ax, vs.b, fill(vs.logE[jE], length(vs.b)), vs.Z[:, jE];
                color = :black, linewidth = 0.6, alpha = 0.5)
        end
        # Curvas por cada b (dirección transversal)
        for ib in 1:4:length(vs.b)   # cada 4 para no saturar
            lines!(ax, fill(vs.b[ib], length(vs.logE)), vs.logE, vs.Z[ib, :];
                color = :black, linewidth = 0.4, alpha = 0.4)
        end

        zlims!(ax, z_floor, 1.06 * zmax_data)
        Colorbar(fig[1,2]; colormap = cmap,
                 limits = (zmin_data, zmax_data), label = zlabel)

        fpath = joinpath(fdir, @sprintf("frame_%03d.png", az_deg))
        save(fpath, fig, px_per_unit = DPI / 96)
        push!(frame_paths, fpath)
        empty!(fig)
    end

    println("    → frames_$tag/ ($(length(frame_paths)) frames)")
    GC.gc(true)

    if GEN_3D_GIF && !isempty(frame_paths)
        gif_path = joinpath(out_dir, "map_3D_$(tag).gif")
        try
            list_file = joinpath(fdir, "filelist.txt")
            open(list_file, "w") do io
                for fp in frame_paths
                    println(io, "file '$(basename(fp))'")
                    println(io, "duration 0.083")
                end
                println(io, "file '$(basename(frame_paths[end]))'")
            end
            run(Cmd(`ffmpeg -y -f concat -safe 0 -i filelist.txt -vf "scale=800:-1" -loop 0 $gif_path`; dir = fdir))
            rm(list_file; force = true)
            println("    → map_3D_$(tag).gif")
        catch e
            println("    ⚠ ffmpeg: $e")
        end
    end
end

"""Genera todos los mapas 2D, superficies 3D y frames."""
function plot_all_maps(rows, energies, out)
    length(energies) < MIN_ENERGIES_MAP && return

    println("\nGenerando mapas 2D / 3D...")
    maps = build_map_matrices(rows, energies)
    d2d = joinpath(out, "mapas_2D")
    d3d = joinpath(out, "mapas_3D")

    map_list = [
        (maps.bPcap,   "b·Pcap",    "bPcap",  "b·Pcap(b, E)",      :capture),
        (maps.bPion,   "b·Pion",    "bPion",  "b·Pion(b, E)",      :ionize),
        (maps.Pcap,    "Pcap",      "Pcap",   "Pcap(b, E)",        :capture),
        (maps.Pion,    "Pion",      "Pion",   "Pion(b, E)",        :ionize),
        (maps.norm,    "‖ψ‖²",     "norm",   "Norma final(b, E)", :both),
        (maps.dE_proj, "|ΔE_proj|", "dEproj", "|ΔE_proj|(b, E)",  :both),
    ]

    for (Z, zl, tag, ttl, mode) in map_list
        want(mode) || continue
        cmap = tag == "norm" ? cgrad(:bwr, rev=true) : MAP_COLORMAP #:bwr :seismic  

        if GEN_MAPS_2D
            println("  Mapa 2D: $tag")
            plot_map2d(Z, maps.b, maps.E, zl, ttl, d2d, "map2d_$tag";cmap=cmap)
        end
        if GEN_MAPS_3D
            println("  Superficie 3D: $tag")
            plot_surf3d(Z, maps.b, maps.E, zl, ttl, d3d, "surf3d_$tag"; cmap=cmap)
        end
        if GEN_3D_FRAMES
            println("  Frames 3D: $tag")
            plot_3d_frames(Z, maps.b, maps.E, zl, ttl, d3d, tag; cmap=cmap)
        end
    end
end

# ════════════════════════════════════════════════════════════════
#  EJECUCIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════

function main()
    set_thesis_theme!()
    out = ensure_dir(joinpath(OUT_DIR, "viz_" * Dates.format(now(), "yyyy-mm-dd_HHMMSS")))

    mode_str = PHYSICS_MODE == :capture ? "Captura" :
               PHYSICS_MODE == :ionize  ? "Ionización" : "Ambos"

    println("═"^62)
    println("  ENDview.jl v$VERSION — $mode_str")
    println("  Raíz: $DATA_ROOT")
    println("  Salida: $out")
    println("═"^62)

    # ── Recolectar ──
    println("\nRecolectando datos...")
    rows, energies = collect_data(DATA_ROOT)
    isempty(rows) && (println("  Sin datos."); return)

    # ── Tablas ──
    println("\nTablas...")
    write_tables(rows, energies, out)

    elist = isempty(E_SELECT) ? energies : filter(E -> E in energies, E_SELECT)

    # ── Curvas por energía ──
    if GEN_CURVES
        println("\nCurvas por energía...")
        for E in elist
            println("  E = $(Elab(E))")
            plot_curves(rows, E, out)
        end
    end

    # ── Overlay ──
    if GEN_OVERLAY && length(energies) >= 2
        println("\nOverlay multi-energía...")
        plot_overlay(rows, energies, out)
    end

    # ── σ(E) y stopping ──
    if (GEN_SIGMA || GEN_STOPPING) && length(energies) >= 2
        println("\nSecciones eficaces y stopping...")
        cs = compute_cross_sections(rows, energies)
        if GEN_SIGMA
            want(:capture) && plot_sigma_cap(cs, out)
            want(:ionize)  && plot_sigma_ion(cs, out)
        end
        if GEN_STOPPING
            plot_stopping(cs, out)
            write_stopping_table(cs, out)
        end
    end

    # ── Mapas 2D / 3D / frames ──
    if GEN_MAPS_2D || GEN_MAPS_3D || GEN_3D_FRAMES
        plot_all_maps(rows, energies, out)
    end

    # ── Plots individuales ──
    if GEN_DENSITY || GEN_ENERGY || GEN_TRAJECTORY
        println("\nPlots individuales...")
        for E in elist
            sub = data_E(rows, E)
            for r in sub
                !isdir(r.sim_dir) && continue
                GEN_DENSITY    && plot_dens(r, out)
                GEN_ENERGY     && plot_ener(r, out)
                GEN_TRAJECTORY && plot_traj(r, out)
                GC.gc(true)   # forzar recolección completa entre simulaciones
            end
        end
    end

    # ── Fin ──
    println("\n" * "═"^62)
    @printf("  Listo. %d energías, %d filas.\n", length(energies), length(rows))
    println("  Figuras en: $out")
    println("═"^62)
end

# ════════════════════════════════════════════════════════════════
main()
