get_limits(px::AbstractVector{<:Real}, x::AbstractMatrix{<:Real}) =
    (min(px[1], minimum(x)), max(px[end], maximum(x)))

# function visualize_data(data, p, x, g)
#     T = p.period
#     (~, B, N) = size(x)
#     t = range(start = 0, step = T, length = N)
#     fig = Figure()
#     ax = [
#         Axis(fig[1, 1]),
#         Axis(fig[2, 1]),
#         Axis(fig[3, 1]),
#         Axis(fig[1:4, 2][1, 1], aspect = DataAspect()),
#         Axis(fig[1:4, 2][2, 1], aspect = DataAspect()),
#     ]

#     for i = 1:B, j = 1:3
#         lines!(ax[j], t, view(x, j, i, :))
#     end

#     sl_x = Slider(fig[4, 1], range = 1:N, startvalue = 1)

#     frame1 = lift(sl_x.value) do x
#         view(g, :, :, x)
#     end

#     frame2 = lift(sl_x.value) do x
#         view(data, :, :, x)
#     end

#     heatmap!(ax[4], frame1, colormap = :grays)
#     heatmap!(ax[5], frame2, colormap = :grays)

#     hidedecorations!.(ax[1:2])
#     hidedecorations!.(ax[4:5])

#     linkxaxes!(ax[1], ax[2], ax[3])

#     xlims!(ax[3], 0, p.period * N)

#     uls = (
#         1.1 * max(maximum(view(x, 1, :, :)), p.pxboundsx[end] - p.pxboundsx[1]),
#         1.1 * max(maximum(view(x, 2, :, :)), p.pxboundsy[end] - p.pxboundsy[1]),
#         1.1 * maximum(view(x, 3, :, :)),
#     )
#     lls = (
#         1.1 * min(minimum(view(x, 1, :, :)), 0),
#         1.1 * min(minimum(view(x, 2, :, :)), 0),
#         1.1 * minimum(view(x, 3, :, :)),
#     )

#     d =
#         max.(
#             uls .- (p.pxnumx, p.pxnumy, 0) .* (p.pxsize / 2),
#             (p.pxnumx, p.pxnumy, 0) .* (p.pxsize / 2) .- lls,
#         )
#     ylims!(
#         ax[1],
#         (p.pxboundsx[end] - p.pxboundsx[1]) / 2 - d[1],
#         (p.pxboundsx[end] - p.pxboundsx[1]) / 2 + d[1],
#     )
#     ylims!(
#         ax[2],
#         (p.pxboundsy[end] - p.pxboundsy[1]) / 2 - d[2],
#         (p.pxboundsy[end] - p.pxboundsy[1]) / 2 + d[2],
#     )
#     ylims!(ax[3], -d[3], d[3])

#     colgap!(fig.layout, 1, 10)
#     colsize!(fig.layout, 2, Relative(1 / 5))
#     rowgap!(fig.layout, 1, 0)
#     rowgap!(fig.layout, 2, 0)

#     return fig
# end

function visualize(v::Video{FT}, gt::Sample{FT}) where {FT}
    if isa(v.data, CuArray)
        to_cpu!(v)
    end
    data = v.data
    p = v.param
    x = gt.x
    B = size(x, 2)

    g = get_px_PSF(gt.x, p.pxboundsx, p.pxboundsy, p.PSF)

    # t = range(start = 0, step = p.period, length = p.length)
    t = 1:p.length
    fig = Figure()
    ax = [
        Axis3(fig[1:3, 1], zlabel = "t"),
        Axis(fig[4, 1], xticks = 0:25:300, xlabel = "t", ylabel = "z"),
        Axis(fig[1:4, 2], aspect = DataAspect()),
    ]

    for m = 1:B
        lines!(ax[1], view(x, 1, m, :), view(x, 2, m, :), t)
        lines!(ax[2], t, view(x, 3, m, :))
    end

    sl_x = Slider(fig[5, 1], range = 1:p.length, startvalue = 1)

    frame1 = lift(sl_x.value) do x
        view(g, :, :, x)
    end

    frame2 = lift(sl_x.value) do x
        view(data, :, :, x)
    end

    f = lift(sl_x.value) do x
        x
    end

    collected_frame = dropdims(sum(data, dims = 3), dims = 3)

    hm = heatmap!(ax[1], p.pxboundsx, p.pxboundsy, frame1, colormap = (:grays, 0.7))

    vl = vlines!(ax[2], t[1])

    on(sl_x.value) do n
        translate!(hm, 0, 0, t[n])
        translate!(vl, t[n], 0, -0.1)
    end

    heatmap!(
        ax[3],
        p.pxboundsx,
        p.pxboundsy .+ (4 * p.pxsize + 2 * p.pxboundsy[end]),
        frame1,
        colormap = :grays,
    )
    translate!(
        text!(
            ax[3],
            p.pxboundsx[1],
            p.pxboundsy[end] + (4 * p.pxsize + 1 * p.pxboundsy[end]),
            text = "asdasd",
            fontsize = 20,
        ),
        0,
        0,
        0.1,
    )

    # text!(ax[2], 0, 0.15, text = string(sl_x.value[]), fontsize = 20)
    heatmap!(
        ax[3],
        p.pxboundsx,
        p.pxboundsy .+ (2 * p.pxsize + p.pxboundsy[end]),
        frame2,
        colormap = :grays,
        colorrange = (false, true),
    )
    heatmap!(
        ax[3],
        p.pxboundsx,
        p.pxboundsy,
        collected_frame,
        colormap = :grays,
        colorrange = (0, maximum(collected_frame)),
    )

    hidexdecorations!(ax[1], label = false)
    hideydecorations!(ax[1], label = false)
    hidedecorations!(ax[3])
    hidespines!(ax[3])

    (lowerx, upperx) = get_limits(p.pxboundsx, view(x, 1, :, :))
    (lowery, uppery) = get_limits(p.pxboundsy, view(x, 2, :, :))

    xlims!(ax[1], lowerx, upperx)
    ylims!(ax[1], lowery, uppery)
    zlims!(ax[1], t[1], t[end])

    xlims!(ax[2], 0, t[end])

    colgap!(fig.layout, 1, 10)
    colsize!(fig.layout, 2, Relative(1 / 3))
    rowgap!(fig.layout, 1, 0)
    rowgap!(fig.layout, 2, 0)
    rowsize!(fig.layout, 4, Relative(1 / 4))
    display(fig)
end

function my_theme()
    Theme(
        Axis = (
            rightspinevisible = false,
            topspinevisible = false,
            xgridvisible = false,
            xticksize = 1,
            ygridvisible = false,
            yticksize = 1,
        ),
        Colorbar = (
            colormap = Reverse(:devon),
            size = 3,
            ticks = (-5:0, ["10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰"]),
            ticksize = 1,
        ),
        # Heatmap = (colormap = Reverse(ColorSchemes.devon)),
        Hist = (normalization = :pdf,),
        Text = (align = (:left, :bottom), font = "Arial", fontsize = 7),
        VLines = (color = ColorSchemes.tab10[2],),
    )
end

function trajcount(M::AbstractMatrix{<:Real}, y::AbstractArray{<:Real})
    counts = Matrix{Int64}(undef, length(y), size(M, 2))
    yedges = Vector{eltype(y)}(undef, length(y) + 1)
    yedges[2:end-1] = (y[1:end-1] + y[2:+end]) / 2
    yedges[1] = 2 * y[1] - yedges[2]
    yedges[end] = 2 * y[end] - yedges[end-1]
    @inbounds for i in axes(M, 2), j in eachindex(y)
        counts[j, i] = count(yedges[j+1] .> view(M, :, i) .>= yedges[j])
    end
    return transpose(counts)
end

function visualize(
    v::Video{FT},
    gt::Sample{FT},
    s::AbstractVector{Sample{FT}};
    num_grid::Integer = 500,
) where {FT}
    if isa(v.data, CuArray)
        to_cpu!(v)
    end
    histcolor =
        RGBAf(ColorSchemes.tab10[1].r, ColorSchemes.tab10[1].g, ColorSchemes.tab10[1].b, 1)
    # fig = Figure(resolution = 72 .* (6, 6), fontsize = 7, font = "Arial")
    set_theme!(my_theme())
    fig = Figure()
    ax = [
        Axis(fig[1, 1][1, 1], title = "x trajectory"),
        Axis(fig[2, 1][1, 1], xlabel = "Frame", title = "y trajectory"),
        Axis(fig[1, 3], xlabel = "Diffusion coefficient (μm²/s)"),
        Axis(fig[1, 2], xlabel = "Resolution (nm)"),
    ]

    all_trajectories = get_x(s)
    x = view(all_trajectories, :, :, 1)
    y = view(all_trajectories, :, :, 2)
    z = view(all_trajectories, :, :, 3)
    t = collect(1:v.param.length)

    B = size(x, 1)
    xrange = range(minimum(x), maximum(x), num_grid)
    yrange = range(minimum(y), maximum(y), num_grid)
    xcount = trajcount(x, xrange) ./ B
    ycount = trajcount(y, yrange) ./ B

    maxcount = log10(max(maximum(xcount), maximum(ycount)))
    mincount = min(
        minimum(replace(log10.(xcount), -Inf => Inf)),
        minimum(replace(log10.(ycount), -Inf => Inf)),
    )

    hm = heatmap!(
        ax[1],
        t,
        xrange,
        replace(log10.(xcount), -Inf => NaN),
        colorrange = (mincount, maxcount),
        colormap = Reverse(ColorSchemes.devon),
    )
    hm2 = heatmap!(
        ax[2],
        t,
        yrange,
        replace(log10.(ycount), -Inf => NaN),
        colorrange = (mincount, maxcount),
        colormap = Reverse(ColorSchemes.devon),
    )

    translate!(hm, 0, 0, -0.2)
    translate!(hm2, 0, 0, -0.2)

    # Colorbar(fig[0, 1], hm2, vertical = false)

    for j = 1:get_B(gt)
        lines!(
            ax[1],
            t,
            view(gt.x, 1, j, :),
            color = ColorSchemes.tab10[2],
            linewidth = 1.5,
        )
        lines!(
            ax[2],
            t,
            view(gt.x, 2, j, :),
            color = ColorSchemes.tab10[2],
            linewidth = 1.5,
        )
    end
    # for j = 1:size(result.X_MAP, 1)
    #     lines!(ax[1], t, result.X_MAP[j, :]; color = ColorSchemes.tab10[3], linewidth = 0.5)
    #     lines!(ax[2], t, result.Y_MAP[j, :]; color = ColorSchemes.tab10[3], linewidth = 0.5)
    # end

    rb = rangebars!(ax[1], [20], [xrange[end] - 0.25], [xrange[end]])
    # txt = text!(ax[1], 0.15, xrange[1] + 0.05, text = "0.25 μm")
    translate!(rb, 0, 0, -0.1)
    # translate!(txt, 0, 0, -0.1)

    rb = rangebars!(ax[2], [20], [yrange[end] - 0.25], [yrange[end]])
    # txt = text!(ax[2], 0.15, yrange[1] + 0.05, text = "0.25 μm")
    translate!(rb, 0, 0, -0.1)
    # translate!(txt, 0, 0, -0.1)

    # D = get_D(s)
    # D_CI = quantile(D, [0.025, 0.5, 0.975])
    # vspan!(ax[3], D_CI[1], D_CI[3], color = :grey80)
    # hist!(ax[3], D, color = histcolor)
    # D_range = range(minimum(D), maximum(D); length = 200)
    # D_prior = InverseGamma(result.D_prior[1], result.D_prior[2])
    # lines!(
    #     ax[3],
    #     D_range,
    #     pdf.(D_prior, D_range),
    #     color = ColorSchemes.tab10[4],
    #     linewidth = 1,
    # )
    # vlines!(ax[3], gt.D, color = ColorSchemes.tab10[2], linewidth = 1)

    # hist!(
    #     ax[6],
    #     localization_errors .* 1000,
    #     color = ColorSchemes.tab10[10],
    #     normalization = :probability,
    # )
    # B_range = range(minimum(result.B), maximum(result.B); length = 200)
    # B_prior = Binomial(100, 0.05)
    # # lines!(ax[4], B_range, pdf.(h_prior, h_range), color = ColorSchemes.tab10[4], linewidth = 1)
    # vlines!(ax[4], 3, color = ColorSchemes.tab10[2], linewidth = 1)
    # vlines!(ax[4], result.h_MAP, color = ColorSchemes.tab10[3], linewidth = 1)

    # F_CI = quantile(result.F, [0.025, 0.5, 0.975])
    # vspan!(ax[5], F_CI[1], F_CI[3], color = :grey80)
    # hist!(ax[5], result.F, color = histcolor)
    # F_range = range(9, 11; length = 200)
    # F_prior = Gamma(result.F_prior[1], result.F_prior[2])
    # lines!(
    #     ax[5],
    #     F_range,
    #     pdf.(F_prior, F_range),
    #     color = ColorSchemes.tab10[4],
    #     linewidth = 1,
    # )
    # vlines!(ax[5], result.F_gnd, color = ColorSchemes.tab10[2], linewidth = 1)
    # xlims!(ax[5], 9.2, 10.7)
    # vlines!(ax[5], result.F_MAP, color = ColorSchemes.tab10[3], linewidth = 1)

    hideydecorations!.(ax[1:2])
    # ylims!(ax[1], limits[1], limits[2])
    # ylims!.(ax[3:6], 0, nothing)

    # elem_1 =
    #     [LineElement(color = ColorSchemes.tab10[2], linestyle = nothing, linewidth = 0.75)]
    # elem_2 =
    #     [LineElement(color = ColorSchemes.tab10[3], linestyle = nothing, linewidth = 0.75)]
    # elem_3 =
    #     [LineElement(color = ColorSchemes.tab10[4], linestyle = nothing, linewidth = 0.75)]
    # elem_4 = [PolyElement(color = ColorSchemes.tab10[1])]
    # elem_5 = [PolyElement(color = :grey80, strokevisible = false)]
    # elem_6 = [PolyElement(color = ColorSchemes.tab10[10])]

    # Legend(
    #     fig[0, 2:3],
    #     [elem_1, elem_2, elem_3, elem_4, elem_5, elem_6],
    #     ["Ground truth", "MAP", "Prior", "Posterior density", "95% CI", "Probability"],
    #     framevisible = false,
    #     patchsize = (15, 15),
    #     nbanks = 3,
    # )

    # for (label, layout) in zip(
    #     ["a", "b", "c", "d", "e", "f", "g"],
    #     [fig[1, 1], fig[2, 1], fig[1, 2], fig[1, 3], fig[2, 2], fig[2, 3], fig[3, :]],
    # )
    #     Label(
    #         layout[1, 1, TopLeft()],
    #         label,
    #         fontsize = 8,
    #         font = "Arial Bold",
    #         padding = (0, 0, 0, 0),
    #     )
    # end

    # colgap!(fig.layout, Relative(0.05))
    # # colgap!(ga, 1, 2)
    # # colgap!(gb, 1, 2)
    colsize!(fig.layout, 1, Relative(2 / 5))
    colsize!(fig.layout, 2, Relative(3 / 10))
    colsize!(fig.layout, 3, Relative(3 / 10))

    # rowsize!(fig.layout, 0, Relative(1 / 10))
    # rowgap!(fig.layout, Relative(0.05))

    # save("test.png", fig, pt_per_unit = 1)
    display(fig)
    set_theme!()
end