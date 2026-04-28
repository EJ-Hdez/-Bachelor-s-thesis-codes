# ============================================================
# RadialPart-H.jl
# Parte radial del átomo de hidrógeno:
#   - Solución numérica (Diferencias Finitas) vs analítica
#   - Graficas en PDF (vectoriales): alta calidad para tesis
#
# Salidas:
#   H_1s_u2.pdf
#   H_2p_u2.pdf
#   H_panel_1s_2p.pdf
#   H_panel_n2.pdf
#   H_panel_n3.pdf
# ============================================================

using LinearAlgebra
using Printf
using CairoMakie

# -----------------------------
# Laguerre asociado: L_k^{α}(x)
# -----------------------------
function assoc_laguerre(k::Int, α::Int, x::Real)
    if k == 0
        return 1.0
    elseif k == 1
        return 1.0 + α - x
    else
        Lkm1 = 1.0
        Lk   = 1.0 + α - x
        for j in 1:(k-1)
            Lkp1 = ((2*j + 1 + α - x)*Lk - (j + α)*Lkm1) / (j + 1)
            Lkm1, Lk = Lk, Lkp1
        end
        return Lk
    end
end

# -----------------------------
# u_analítico(r) = r R_{nl}(r)
# (unidades atómicas por default: a0=1)
# -----------------------------
function u_analytic(r::Real, n::Int, l::Int; Z::Real=1.0, a0::Real=1.0)
    @assert n ≥ 1 && l ≥ 0 && n > l "Se requiere n>l≥0"
    ρ = 2Z*r/(n*a0)
    k = n - l - 1
    α = 2l + 1

    # Normalización estándar de R_{nl} (hidrogenoide)
    num = factorial(big(k))
    den = 2n * factorial(big(n + l))
    N  = (2Z/(n*a0))^(3/2) * sqrt(Float64(num)/Float64(den))

    R = N * exp(-ρ/2) * ρ^l * assoc_laguerre(k, α, ρ)
    return r * R
end

# -----------------------------
# Hamiltoniano radial para u(r)=rR(r)
#   [-1/(2μ) d²/dr² + l(l+1)/(2μ r²) - Z/r] u = E u
# Condiciones:
#   u(0)=0 y u(rmax)=0 (frontera tipo caja numérica)
# -----------------------------
function radial_hamiltonian(dr::Real, rmax::Real, l::Int; Z::Real=1.0, μ::Real=1.0)
    nfull = Int(round(rmax/dr))
    @assert abs(nfull*dr - rmax) < 1e-12 "Elige rmax múltiplo exacto de dr"

    nr = nfull - 1               # puntos interiores (excluye r=0 y r=rmax)
    r  = dr .* collect(1:nr)     # r_i = i dr

    off  = fill(-1/(2μ*dr^2), nr-1)
    diag = fill( 1/(μ*dr^2),  nr)

    Veff = @. l*(l+1)/(2μ*r^2) - Z/r
    diag .+= Veff

    return r, SymTridiagonal(diag, off)
end

# Normalización: ∫|u|^2 dr = 1
function normalize_u!(u::AbstractVector, dr::Real)
    u ./= sqrt(sum(abs2, u) * dr)
    return u
end

# <r> usando u(r):  <r> = ∫ r |u(r)|^2 dr
function expectation_r(r::AbstractVector, u::AbstractVector, dr::Real)
    return sum(r .* abs2.(u)) * dr
end

# Para un l fijo, el k-ésimo estado ligado corresponde a n = l + k
function solve_state(dr, rmax, n, l; Z=1.0, μ=1.0)
    @assert n > l "Se requiere n>l"
    r, H = radial_hamiltonian(dr, rmax, l; Z=Z, μ=μ)
    F = eigen(H)                # tridiagonal simétrica
    k = n - l                   # k=1 -> n=l+1
    E = F.values[k]
    u = copy(F.vectors[:, k])
    normalize_u!(u, dr)
    return r, E, u
end

function analytic_on_grid(r, dr, n, l; Z=1.0, a0=1.0)
    u = [u_analytic(ri, n, l; Z=Z, a0=a0) for ri in r]
    normalize_u!(u, dr)
    return u
end

# ============================================================
# Gráfica individual: |u(r)|^2 (analítica + numérica)
# ============================================================
function plot_u2_single(r, u_num, u_ana;
                        title="",
                        outfile="radial.pdf",
                        ana_color=:blue,
                        num_color=:red,
                        plot_every::Int=15,
                        markersize=7,
                        linewidth=1.5,
                        fig_size=(1100, 520))

    fig = Figure(size=fig_size)
    ax  = Axis(fig[1,1], xlabel="r (a.u.)", ylabel="|u(r)|²", title=title)

    yA = abs2.(u_ana)
    yN = abs2.(u_num)
    idx = 1:plot_every:length(r)

    # Marcadores: numérico (submuestreo visual)
    scatter!(ax, r[idx], yN[idx];
             marker=:x, markersize=markersize, color=num_color,
             label="Numérica (FD)")

    # Línea: analítica
    lines!(ax, r, yA;
           linewidth=linewidth, color=ana_color,
           label="Analítica")

    axislegend(ax, position=:rt, framevisible=true, labelsize=12, patchsize=(26, 14))
    save(outfile, fig)
end

# ============================================================
# Gráfica "panel" (en realidad: una figura por panel)
# Cada estado: misma paleta (línea analítica + marcadores FD del mismo color)
# Leyenda: SOLO (n,l) para no duplicar entradas (el estilo se explica en el texto del plot).
# ============================================================
function plot_u2_panel(dr, rmax, states;
                       title="",
                       outfile="panel.pdf",
                       plot_every::Int=25,
                       markersize=7,
                       linewidth=1.5,
                       fig_size=(1100, 520))

    fig = Figure(size=fig_size)
    ax  = Axis(fig[1,1], xlabel="r (a.u.)", ylabel="|u(r)|²", title=title)

    colors  = Makie.wong_colors()
    markers = [:x, :circle, :utriangle, :diamond, :star5, :cross, :rect, :plus]

    for (i, (n,l)) in enumerate(states)
        r, E, uN = solve_state(dr, rmax, n, l)
        uA = analytic_on_grid(r, dr, n, l)

        col = colors[1 + (i-1) % length(colors)]
        mkr = markers[1 + (i-1) % length(markers)]
        idx = 1:plot_every:length(r)

        # Numérico: marcadores (sin label)
        scatter!(ax, r[idx], abs2.(uN[idx]);
                 color=col, marker=mkr, markersize=markersize)

        # Analítico: línea (con label)
        lines!(ax, r, abs2.(uA);
               color=col, linewidth=linewidth, label="n=$n, l=$l")
    end

    axislegend(ax, position=:rt, framevisible=true, labelsize=12, patchsize=(26, 14))
    save(outfile, fig)
end

# ============================================================
# Demos (1s, 2p + paneles separados)
# ============================================================
function main()
    # Parámetros (a.u.)
    dr = 0.01

    # -------------------------
    # 1s (rmax=7)
    # -------------------------
    n1, l1 = 1, 0
    r1, E1, u1N = solve_state(dr, 7.0, n1, l1)
    u1A = analytic_on_grid(r1, dr, n1, l1)

    E1_exact = -1/(2n1^2)
    r1_exact = 0.5*(3n1^2 - l1*(l1+1))
    r1_num   = expectation_r(r1, u1N, dr)

    @printf("1s (n=%d,l=%d):\n", n1, l1)
    @printf("  E_num   = %.15f a.u.\n", E1)
    @printf("  E_exact = %.15f a.u.\n", E1_exact)
    @printf("  rel.err(E) = %.6f %%\n", 100*abs((E1 - E1_exact)/E1_exact))
    @printf("  <r>_num   = %.15f a.u.\n", r1_num)
    @printf("  <r>_exact = %.15f a.u.\n", r1_exact)
    @printf("  rel.err(<r>) = %.6f %%\n\n", 100*abs((r1_num - r1_exact)/r1_exact))

    plot_u2_single(r1, u1N, u1A;
        title="Hidrógeno 1s: |u(r)|²  (Δr=0.01, rmax=7.0)",
        outfile="H_1s_u2.pdf",
        plot_every=15)

    # -------------------------
    # 2p (rmax=30)
    # -------------------------
    n2, l2 = 2, 1
    r2, E2, u2N = solve_state(dr, 30.0, n2, l2)
    u2A = analytic_on_grid(r2, dr, n2, l2)

    E2_exact = -1/(2n2^2)
    r2_exact = 0.5*(3n2^2 - l2*(l2+1))
    r2_num   = expectation_r(r2, u2N, dr)

    @printf("2p (n=%d,l=%d):\n", n2, l2)
    @printf("  E_num   = %.15f a.u.\n", E2)
    @printf("  E_exact = %.15f a.u.\n", E2_exact)
    @printf("  rel.err(E) = %.6f %%\n", 100*abs((E2 - E2_exact)/E2_exact))
    @printf("  <r>_num   = %.15f a.u.\n", r2_num)
    @printf("  <r>_exact = %.15f a.u.\n", r2_exact)
    @printf("  rel.err(<r>) = %.6f %%\n\n", 100*abs((r2_num - r2_exact)/r2_exact))

    plot_u2_single(r2, u2N, u2A;
        title="Hidrógeno 2p: |u(r)|²  (Δr=0.01, rmax=30.0)",
        outfile="H_2p_u2.pdf",
        plot_every=25)

    # ---------------------------------------------------------
    # Paneles separados (todos con rmax=30 para mismo eje x)
    # ---------------------------------------------------------
    plot_u2_panel(dr, 30.0, [(1,0), (2,1)];
        title="Hidrógeno: 1s y 2p  (línea analítica + marcadores FD)",
        outfile="H_panel_1s_2p.pdf",
        plot_every=30)

    plot_u2_panel(dr, 30.0, [(2,0), (2,1)];
        title="Hidrógeno: n = 2  (2s, 2p)  (línea analítica + marcadores FD)",
        outfile="H_panel_n2.pdf",
        plot_every=30)

    plot_u2_panel(dr, 30.0, [(3,0), (3,1), (3,2)];
        title="Hidrógeno: n = 3  (3s, 3p, 3d)  (línea analítica + marcadores FD)",
        outfile="H_panel_n3.pdf",
        plot_every=30)

    println("Listo: PDFs generados.")
end

main()
