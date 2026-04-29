##############################
# Paquete gaussiano 3D libre #
# Crank–Nicolson (ADI) real  #
# Animaciones 3D y en la tapa#
##############################

using LinearAlgebra
using GLMakie
using Dates

#----------------------------------------------------------
# Normalización: ∫ |ψ|^2 dV = 1
#----------------------------------------------------------
function normalize!(ψ::Array{ComplexF64,3}, dx, dy, dz)
    norm2 = 0.0
    Nx, Ny, Nz = size(ψ)
    @inbounds for i in 1:Nx, j in 1:Ny, k in 1:Nz
        norm2 += abs2(ψ[i,j,k])
    end
    norm2 *= dx * dy * dz
    ψ ./= sqrt(norm2)
    return norm2
end

function norm2(ψ::Array{ComplexF64,3}, dx, dy, dz)
    s = 0.0
    Nx, Ny, Nz = size(ψ)
    @inbounds for i in 1:Nx, j in 1:Ny, k in 1:Nz
        s += abs2(ψ[i,j,k])
    end
    return s * dx * dy * dz
end

#----------------------------------------------------------
# Valores esperados <x>, <y>, <z> y segundo momento en z
#----------------------------------------------------------
function expectation_xyz(
    ψ::Array{ComplexF64,3},
    x::AbstractVector, y::AbstractVector, z::AbstractVector,
    dx, dy, dz
)
    Nx, Ny, Nz = size(ψ)
    w  = dx * dy * dz
    ex = 0.0
    ey = 0.0
    ez = 0.0

    @inbounds for i in 1:Nx, j in 1:Ny, k in 1:Nz
        ρ = abs2(ψ[i,j,k])
        ex += x[i] * ρ
        ey += y[j] * ρ
        ez += z[k] * ρ
    end

    ex *= w
    ey *= w
    ez *= w

    return ex, ey, ez
end

function second_moment_z(
    ψ::Array{ComplexF64,3},
    z::AbstractVector,
    dx, dy, dz
)
    Nx, Ny, Nz = size(ψ)
    w   = dx * dy * dz
    ez2 = 0.0

    @inbounds for i in 1:Nx, j in 1:Ny, k in 1:Nz
        ρ = abs2(ψ[i,j,k])
        ez2 += z[k]^2 * ρ
    end

    return ez2 * w
end

#----------------------------------------------------------
# Extraer puntos para el scatter 3D a partir de |ψ|²
#----------------------------------------------------------
function extract_points(
    ψ::Array{ComplexF64,3},
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector;
    threshold_fraction = 0.03
)
    Nx, Ny, Nz = size(ψ)

    ρmax = maximum(abs2, ψ)
    thresh = threshold_fraction * ρmax

    xs = Float32[]
    ys = Float32[]
    zs = Float32[]
    cs = Float32[]

    @inbounds for i in 1:Nx, j in 1:Ny, k in 1:Nz
        val = abs2(ψ[i,j,k])
        if val > thresh
            push!(xs, x[i])
            push!(ys, y[j])
            push!(zs, z[k])
            push!(cs, val)
        end
    end

    return xs, ys, zs, cs
end

#----------------------------------------------------------
# Barridos unidimensionales (ADI) con buffers reutilizables
#----------------------------------------------------------
function sweep_x!(
    tmp1::Array{ComplexF64,3},
    f::Array{ComplexF64,3},
    Ax_fact, Ax_minus
)
    Nx, Ny, Nz = size(f)
    Nx_int = Nx - 2
    line = Vector{ComplexF64}(undef, Nx_int)
    rhs  = similar(line)
    sol  = similar(line)

    @inbounds for j in 1:Ny, k in 1:Nz
        @views copyto!(line, f[2:Nx-1, j, k])
        mul!(rhs, Ax_minus, line)
        ldiv!(sol, Ax_fact, rhs)

        tmp1[1,  j, k] = 0.0 + 0.0im
        tmp1[Nx, j, k] = 0.0 + 0.0im
        @views tmp1[2:Nx-1, j, k] .= sol
    end
end

function sweep_y!(
    tmp2::Array{ComplexF64,3},
    tmp1::Array{ComplexF64,3},
    Ay_fact, Ay_minus
)
    Nx, Ny, Nz = size(tmp1)
    Ny_int = Ny - 2
    line = Vector{ComplexF64}(undef, Ny_int)
    rhs  = similar(line)
    sol  = similar(line)

    @inbounds for i in 1:Nx, k in 1:Nz
        @views copyto!(line, tmp1[i, 2:Ny-1, k])
        mul!(rhs, Ay_minus, line)
        ldiv!(sol, Ay_fact, rhs)

        tmp2[i, 1,  k] = 0.0 + 0.0im
        tmp2[i, Ny, k] = 0.0 + 0.0im
        @views tmp2[i, 2:Ny-1, k] .= sol
    end
end

function sweep_z!(
    ψ::Array{ComplexF64,3},
    tmp2::Array{ComplexF64,3},
    Az_fact, Az_minus
)
    Nx, Ny, Nz = size(tmp2)
    Nz_int = Nz - 2
    line = Vector{ComplexF64}(undef, Nz_int)
    rhs  = similar(line)
    sol  = similar(line)

    @inbounds for i in 1:Nx, j in 1:Ny
        @views copyto!(line, tmp2[i, j, 2:Nz-1])
        mul!(rhs, Az_minus, line)
        ldiv!(sol, Az_fact, rhs)

        ψ[i, j, 1]  = 0.0 + 0.0im
        ψ[i, j, Nz] = 0.0 + 0.0im
        @views ψ[i, j, 2:Nz-1] .= sol
    end
end

#----------------------------------------------------------
# Un paso de tiempo Crank–Nicolson 3D (ADI)
#----------------------------------------------------------
function cn_step!(
    ψ::Array{ComplexF64,3}, V::Array{Float64,3},
    Ax_fact, Ax_minus,
    Ay_fact, Ay_minus,
    Az_fact, Az_minus,
    dt::Float64, ħ::Float64,
    f::Array{ComplexF64,3},
    tmp1::Array{ComplexF64,3},
    tmp2::Array{ComplexF64,3}
)
    Nx, Ny, Nz = size(ψ)

    # 1) Patada de potencial
    @inbounds for i in 1:Nx, j in 1:Ny, k in 1:Nz
        f[i,j,k] = exp(-1im * V[i,j,k] * dt / ħ) * ψ[i,j,k]
    end

    # 2–4) Barridos x, y, z
    sweep_x!(tmp1, f,   Ax_fact, Ax_minus)
    sweep_y!(tmp2, tmp1, Ay_fact, Ay_minus)
    sweep_z!(ψ,   tmp2, Az_fact, Az_minus)

    return nothing
end

#----------------------------------------------------------
# Programa principal
#----------------------------------------------------------
function main()
    # --- 0. Carpeta de salida con fecha ---
    base_dir = "/home/ej-hdez/Documentos/Códigos de Programación/Datos y gráficos obtenidos/Gaussiana 3D"
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    sim_name  = "simulacion_" * timestamp
    sim_dir   = joinpath(base_dir, sim_name)
    mkpath(sim_dir)

    video_path_main   = joinpath(sim_dir, "gauss3d_CN_3D.mp4")
    video_path_toplin = joinpath(sim_dir, "tapa_superior_linear.mp4")
    video_path_toplog = joinpath(sim_dir, "tapa_superior_log.mp4")
    info_path         = joinpath(sim_dir, "info_simulacion.txt")

    # Cronómetro
    t_start = time()

    # --- 1. Parámetros físicos (u.a.) ---
    ħ = 1.0
    m = 1.0

    # --- 2. Malla espacial (prisma rectangular) ---
    Lx = 10.0
    Ly = 10.0
    Lz = 40.0

    Nx = 100
    Ny = 100
    Nz = 250

    x = range(-Lx/2, Lx/2; length = Nx)
    y = range(-Ly/2, Ly/2; length = Ny)
    z = range(0.0,  Lz;    length = Nz)

    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    # --- 3. Paso temporal ---
    dt   = 0.02           # paso de tiempo
    Nt   = 600            # número de pasos de tiempo
    obs_skip = 10         # cada cuántos pasos imprimimos observables

    # --- 4. Potencial (libre) ---
    V = zeros(Float64, Nx, Ny, Nz)

    # --- 5. Paquete gaussiano 3D inicial (k0 ≠ 0 en z) ---
    ψ   = zeros(ComplexF64, Nx, Ny, Nz)
    f    = similar(ψ)
    tmp1 = similar(ψ)
    tmp2 = similar(ψ)

    x0 = 0.0
    y0 = 0.0
    z0 = Lz * 0.25    # centro inicial en z

    σx = 1.0
    σy = 1.0
    σz = 1.0

    k0 = 1.0          # momento medio en z (v_g ≈ k0/m)

    @inbounds for i in 1:Nx, j in 1:Ny, k in 1:Nz
        xx = x[i]
        yy = y[j]
        zz = z[k]

        envelope = exp(-(
            (xx - x0)^2/(2σx^2) +
            (yy - y0)^2/(2σy^2) +
            (zz - z0)^2/(2σz^2)
        ))
        phase = exp(1im * k0 * (zz - z0))

        ψ[i, j, k] = envelope * phase
    end

    # Normalización numérica
    norm_before = normalize!(ψ, dx, dy, dz)
    println("Norma ANTES de normalizar ≈ ", norm_before)

    norm_after = norm2(ψ, dx, dy, dz)
    println("Norma DESPUÉS de normalizar ≈ ", norm_after)

    # --- 6. Construcción de matrices CN (sólo interiores) ---
    νx = 1im * ħ * dt / (4m * dx^2)
    νy = 1im * ħ * dt / (4m * dy^2)
    νz = 1im * ħ * dt / (4m * dz^2)

    Nx_int = Nx - 2
    Ny_int = Ny - 2
    Nz_int = Nz - 2

    # A± en x
    main_x_plus  = fill(1 + 2νx, Nx_int)
    off_x_plus   = fill(-νx,     Nx_int - 1)
    main_x_minus = fill(1 - 2νx, Nx_int)
    off_x_minus  = fill(νx,      Nx_int - 1)

    Ax_plus  = Tridiagonal(off_x_plus,  main_x_plus,  off_x_plus)
    Ax_minus = Tridiagonal(off_x_minus, main_x_minus, off_x_minus)
    Ax_fact  = lu(Ax_plus)

    # B± en y
    main_y_plus  = fill(1 + 2νy, Ny_int)
    off_y_plus   = fill(-νy,     Ny_int - 1)
    main_y_minus = fill(1 - 2νy, Ny_int)
    off_y_minus  = fill(νy,      Ny_int - 1)

    Ay_plus  = Tridiagonal(off_y_plus,  main_y_plus,  off_y_plus)
    Ay_minus = Tridiagonal(off_y_minus, main_y_minus, off_y_minus)
    Ay_fact  = lu(Ay_plus)

    # C± en z
    main_z_plus  = fill(1 + 2νz, Nz_int)
    off_z_plus   = fill(-νz,     Nz_int - 1)
    main_z_minus = fill(1 - 2νz, Nz_int)
    off_z_minus  = fill(νz,      Nz_int - 1)

    Az_plus  = Tridiagonal(off_z_plus,  main_z_plus,  off_z_plus)
    Az_minus = Tridiagonal(off_z_minus, main_z_minus, off_z_minus)
    Az_fact  = lu(Az_plus)

    # Números tipo Courant (a partir de νx,νy,νz)
    Cx = ħ * dt / (4m * dx^2)
    Cy = ħ * dt / (4m * dy^2)
    Cz = ħ * dt / (4m * dz^2)
    println("Números tipo Courant: Cx = $Cx,  Cy = $Cy,  Cz = $Cz")

    # Observables iniciales
    ex0, ey0, ez0 = expectation_xyz(ψ, x, y, z, dx, dy, dz)
    ez20          = second_moment_z(ψ, z, dx, dy, dz)
    σz0           = sqrt(ez20 - ez0^2)
    println("t = 0.0,  norma = $(norm2(ψ,dx,dy,dz)),  <z> = $ez0,  σz = $σz0")

    # --- 7. Preparar visualización 3D principal ---
    GLMakie.activate!()

    xs0, ys0, zs0, cs0 = extract_points(ψ, x, y, z; threshold_fraction = 0.03)
    xs_obs = Observable(xs0)
    ys_obs = Observable(ys0)
    zs_obs = Observable(zs0)
    cs_obs = Observable(cs0)

    fig = Figure(size = (900, 700))
    ax  = Axis3(fig[1, 1],
                xlabel = "x (u.a.)",
                ylabel = "y (u.a.)",
                zlabel = "z (u.a.)",
                title  = "Paquete gaussiano 3D libre, t = 0.0")

    scatter!(ax, xs_obs, ys_obs, zs_obs;
             markersize = 8,
             color      = cs_obs,
             colormap   = :plasma)

    # --- 7.1 Buffers para tapa superior ---
    total_steps = Nt
    k_top = Nz - 1                       # último plano interior en z
    top_frames = Vector{Matrix{Float64}}(undef, total_steps + 1)
    global_max_top = 0.0

    # --- 8. Animación principal y recolección de datos ---
    record(fig, video_path_main, 0:total_steps) do n
        t = n * dt

        if n > 0
            cn_step!(ψ, V,
                     Ax_fact, Ax_minus,
                     Ay_fact, Ay_minus,
                     Az_fact, Az_minus,
                     dt, ħ,
                     f, tmp1, tmp2)
        end

        # Scatter general
        xs, ys, zs, cs = extract_points(ψ, x, y, z; threshold_fraction = 0.03)
        xs_obs[] = xs
        ys_obs[] = ys
        zs_obs[] = zs
        cs_obs[] = cs
        ax.title = "Paquete gaussiano 3D libre   t = $(round(t, digits=2))"

        # Densidad en la tapa superior
        ρ_top = abs2.( @view ψ[:, :, k_top] )
        top_frames[n+1] = copy(ρ_top)
        max_local = maximum(ρ_top)
        if max_local > global_max_top
            global_max_top = max_local
        end

        # Observables cada obs_skip pasos
        if n % obs_skip == 0
            nrm = norm2(ψ, dx, dy, dz)
            _, _, ez = expectation_xyz(ψ, x, y, z, dx, dy, dz)
            ez2      = second_moment_z(ψ, z, dx, dy, dz)
            σz       = sqrt(ez2 - ez^2)
            println("t = $(round(t,digits=3)),  norma = $nrm,  <z> = $ez,  σz = $σz")
        end
    end

    println("Película principal guardada en: ", video_path_main)

    # --- 9. Película de la tapa superior (escala lineal) ---
    fig_tlin = Figure(size = (900, 700))
    ax_tlin  = Axis(fig_tlin[1, 1],
                    xlabel = "x (u.a.)",
                    ylabel = "y (u.a.)",
                    title  = "Densidad en tapa superior (lineal)")

    ρ_top_obs = Observable(top_frames[1])

    heatmap!(ax_tlin, x, y, ρ_top_obs;
             colormap = :plasma,
             colorrange = (0.0, global_max_top))

    record(fig_tlin, video_path_toplin, 0:total_steps) do n
        t = n * dt
        ρ_top_obs[] = top_frames[n+1]
        ax_tlin.title = "Densidad en tapa superior (lineal)   t = $(round(t,digits=2))"
    end

    println("Película tapa superior (lineal) guardada en: ", video_path_toplin)

        # --- 10. Película de la tapa superior (escala log10, corte 1e-4) ---
    ρmin_clip = 1e-4

    # Si la máxima densidad en la tapa es muy pequeña, global_max_top puede ser < ρmin_clip
    # En ese caso, usamos ρmin_clip como máximo "visible", pero evitamos rango degenerado.
    max_for_log = max(global_max_top, ρmin_clip)

    log_min = log10(ρmin_clip)
    log_max = log10(max_for_log)

    # Evitar colorrange degenerado (min == max)
    if log_max <= log_min
        log_max = log_min + 1.0   # por ejemplo, de -4 a -3
    end

    fig_tlog = Figure(size = (900, 700))
    ax_tlog  = Axis(fig_tlog[1, 1],
                    xlabel = "x (u.a.)",
                    ylabel = "y (u.a.)",
                    title  = "Densidad en tapa superior (log10)")

    # frame inicial en log
    frame0 = top_frames[1]
    frame0_clip = max.(frame0, ρmin_clip)
    ρ_log_obs = Observable(log10.(frame0_clip))

    heatmap!(ax_tlog, x, y, ρ_log_obs;
             colormap = :plasma,
             colorrange = (log_min, log_max))

    record(fig_tlog, video_path_toplog, 0:total_steps) do n
        t = n * dt
        frame = top_frames[n+1]
        frame_clip = max.(frame, ρmin_clip)
        ρ_log_obs[] = log10.(frame_clip)
        ax_tlog.title = "Densidad en tapa superior (log10)   t = $(round(t,digits=2))"
    end
    println("global_max_top en la tapa = ", global_max_top)
    println("Película tapa superior (log) guardada en: ", video_path_toplog)


    # --- 11. Norma y observables finales + tiempo total ---
    norm_final = norm2(ψ, dx, dy, dz)
    exf, eyf, ezf = expectation_xyz(ψ, x, y, z, dx, dy, dz)
    ez2f          = second_moment_z(ψ, z, dx, dy, dz)
    σzf           = sqrt(ez2f - ezf^2)
    T_final       = Nt * dt

    elapsed = time() - t_start

    println("Norma final ≈ ", norm_final)
    println("Tiempo total de simulación ≈ $(elapsed) s")

    # --- 12. Guardar info en archivo de texto ---
    open(info_path, "w") do io
        println(io, "Simulación paquete gaussiano 3D libre (Crank–Nicolson ADI)")
        println(io, "Fecha y hora: ", timestamp)
        println(io)

        println(io, "# Parámetros físicos")
        println(io, "ħ = $ħ,  m = $m")
        println(io)

        println(io, "# Dominio y malla")
        println(io, "Lx = $Lx,  Ly = $Ly,  Lz = $Lz")
        println(io, "Nx = $Nx,  Ny = $Ny,  Nz = $Nz")
        println(io, "dx = $dx,  dy = $dy,  dz = $dz")
        println(io)

        println(io, "# Paquete inicial")
        println(io, "x0 = $x0,  y0 = $y0,  z0 = $z0")
        println(io, "σx = $σx,  σy = $σy,  σz = $σz")
        println(io, "k0 = $k0")
        println(io)

        println(io, "# Tiempo")
        println(io, "dt = $dt,  Nt = $Nt,  T_final = $T_final")
        println(io)

        println(io, "# Parámetros tipo Courant")
        println(io, "Cx = $Cx,  Cy = $Cy,  Cz = $Cz")
        println(io)

        println(io, "# Resultados")
        println(io, "Norma inicial (ya normalizada) = $norm_after")
        println(io, "Norma final                  = $norm_final")
        println(io, "<z>(0)                       = $ez0")
        println(io, "<z>(T_final)                 = $ezf")
        println(io, "Velocidad promedio ≈ ", (ezf - ez0) / T_final)
        println(io, "σz(0)                        = $σz0")
        println(io, "σz(T_final)                  = $σzf")
        println(io)

        println(io, "# Rendimiento")
        println(io, "Tiempo total de simulación (s) = $elapsed")
        println(io)

        println(io, "# Archivos de salida")
        println(io, "Película principal           = $video_path_main")
        println(io, "Película tapa (lineal)       = $video_path_toplin")
        println(io, "Película tapa (log10)        = $video_path_toplog")
    end

    println("Información de la simulación guardada en: ", info_path)
end

main()
