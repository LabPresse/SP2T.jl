# get_lowerlimit(px_min::Real, x::AbstractMatrix{<:Real}) = min(px_min, minimum(x))
# get_upperlimit(px_max::Real, x::AbstractMatrix{<:Real}) = max(px_max, maximum(x))

get_limits(px::AbstractVector{<:Real}, x::AbstractMatrix{<:Real}) =
    (min(px[1], minimum(x)), max(px[end], maximum(x)))

function visualize_data(data, p, x, g)
    T = p.period
    (~, B, N) = size(x)
    t = range(start=0, step=T, length=N)
    fig = Figure()
    ax = [
        Axis(fig[1, 1]),
        Axis(fig[2, 1]),
        Axis(fig[3, 1]),
        Axis(fig[1:4, 2][1, 1], aspect=DataAspect()),
        Axis(fig[1:4, 2][2, 1], aspect=DataAspect()),
    ]

    for i = 1:B, j = 1:3
        lines!(ax[j], t, view(x, j, i, :))
    end

    sl_x = Slider(fig[4, 1], range=1:N, startvalue=1)

    frame1 = lift(sl_x.value) do x
        view(g, :, :, x)
    end

    frame2 = lift(sl_x.value) do x
        view(data, :, :, x)
    end

    # for i = 1:3
    #     vspan!(ax[i], low, high)
    # end


    heatmap!(ax[4], frame1, colormap=:grays)
    heatmap!(ax[5], frame2, colormap=:grays)

    hidedecorations!.(ax[1:2])
    hidedecorations!.(ax[4:5])

    linkxaxes!(ax[1], ax[2], ax[3])

    xlims!(ax[3], 0, p.period * N)

    uls = (
        1.1 * max(maximum(view(x, 1, :, :)), p.pxboundsx[end] - p.pxboundsx[1]),
        1.1 * max(maximum(view(x, 2, :, :)), p.pxboundsy[end] - p.pxboundsy[1]),
        1.1 * maximum(view(x, 3, :, :)),
    )
    lls = (
        1.1 * min(minimum(view(x, 1, :, :)), 0),
        1.1 * min(minimum(view(x, 2, :, :)), 0),
        1.1 * minimum(view(x, 3, :, :)),
    )

    d =
        max.(
            uls .- (p.pxnumx, p.pxnumy, 0) .* (p.pxsize / 2),
            (p.pxnumx, p.pxnumy, 0) .* (p.pxsize / 2) .- lls,
        )
    ylims!(
        ax[1],
        (p.pxboundsx[end] - p.pxboundsx[1]) / 2 - d[1],
        (p.pxboundsx[end] - p.pxboundsx[1]) / 2 + d[1],
    )
    ylims!(
        ax[2],
        (p.pxboundsy[end] - p.pxboundsy[1]) / 2 - d[2],
        (p.pxboundsy[end] - p.pxboundsy[1]) / 2 + d[2],
    )
    ylims!(ax[3], -d[3], d[3])

    colgap!(fig.layout, 1, 10)
    colsize!(fig.layout, 2, Relative(1 / 5))
    rowgap!(fig.layout, 1, 0)
    rowgap!(fig.layout, 2, 0)

    # limits!(ax, 0, 10, 0, 10)

    return fig
end

function visualize_data_3D(v::Video, s::Sample)
    data = v.data
    p = v.params
    x = s.x
    B = size(x, 2)

    g = Array{get_type(s),3}(undef, p.pxnumx, p.pxnumy, p.length)
    simulate!(g, s.x, p.pxboundsx, p.pxboundsy, p.PSF)

    t = range(start=0, step=p.period, length=p.length)
    fig = Figure()
    ax = [
        Axis3(fig[1:3, 1], zlabel="t"),
        Axis(fig[4, 1], xlabel="t", ylabel="z"),
        Axis(fig[1:4, 2], aspect=DataAspect()),
        # Axis(fig[3:4, 2], aspect = DataAspect()),
    ]

    for m = 1:B
        lines!(ax[1], view(x, 1, m, :), view(x, 2, m, :), t)
        lines!(ax[2], t, view(x, 3, m, :))
    end

    sl_x = Slider(fig[5, 1], range=1:p.length, startvalue=1)

    frame1 = lift(sl_x.value) do x
        view(g, :, :, x)
    end

    frame2 = lift(sl_x.value) do x
        view(data, :, :, x)
    end

    # for i = 1:3
    #     vlines!(ax[2], low, high)
    # end

    hm = heatmap!(ax[1], p.pxboundsx, p.pxboundsy, frame1, colormap=(:grays, 0.7))

    vl = vlines!(ax[2], t[1])

    on(sl_x.value) do n
        translate!(hm, 0, 0, t[n])
        translate!(vl, t[n], 0, -0.1)
    end

    heatmap!(
        ax[3],
        p.pxboundsx,
        p.pxboundsy .+ (2 * p.pxsize + p.pxboundsy[end]),
        frame1,
        colormap=:grays,
    )
    heatmap!(
        ax[3],
        p.pxboundsx,
        p.pxboundsy,
        frame2,
        colormap=:grays,
        colorrange=(false, true),
    )

    hidexdecorations!(ax[1], label=false)
    hideydecorations!(ax[1], label=false)
    hidedecorations!(ax[3])
    # hidedecorations!(ax[3])
    hidespines!(ax[3])

    # linkxaxes!(ax[1], ax[2], ax[3])

    # limits!(
    #     ax[1],
    #     p.pxboundsx[1], p.pxboundsx[end],
    #     p.pxboundsy[1], p.pxboundsy[end],
    # )

    (lowerx, upperx) = get_limits(p.pxboundsx, view(x, 1, :, :))
    (lowery, uppery) = get_limits(p.pxboundsy, view(x, 2, :, :))
    # limits!(ax[1], lowerx, upperx, lowery, uppery)

    xlims!(ax[1], lowerx, upperx)
    ylims!(ax[1], lowery, uppery)
    zlims!(ax[1], t[1], t[end])

    xlims!(ax[2], 0, t[end])

    # uls = (
    #     1.1 * max(
    #         maximum(view(gt.tracks, 1, :, :)),
    #         p.pxboundsx[end] - p.pxboundsx[1],
    #     ),
    #     1.1 * max(
    #         maximum(view(gt.tracks, 2, :, :)),
    #         p.pxboundsy[end] - p.pxboundsy[1],
    #     ),
    #     1.1 * maximum(view(gt.tracks, 3, :, :)),
    # )
    # lls = (
    #     1.1 * min(minimum(view(gt.tracks, 1, :, :)), 0),
    #     1.1 * min(minimum(view(gt.tracks, 2, :, :)), 0),
    #     1.1 * minimum(view(gt.tracks, 3, :, :)),
    # )

    # d =
    #     max.(
    #         uls .- (p.pxnumx, p.pxnumy, 0) .* (p.pxsize / 2),
    #         (p.pxnumx, p.pxnumy, 0) .* (p.pxsize / 2) .- lls,
    #     )
    # ylims!(
    #     ax[1],
    #     (p.pxboundsx[end] - p.pxboundsx[1]) / 2 - d[1],
    #     (p.pxboundsx[end] - p.pxboundsx[1]) / 2 + d[1],
    # )
    # ylims!(
    #     ax[2],
    #     (p.pxboundsy[end] - p.pxboundsy[1]) / 2 - d[2],
    #     (p.pxboundsy[end] - p.pxboundsy[1]) / 2 + d[2],
    # )
    # ylims!(ax[3], -d[3], d[3])

    colgap!(fig.layout, 1, 10)
    colsize!(fig.layout, 2, Relative(1 / 3))
    rowgap!(fig.layout, 1, 0)
    rowgap!(fig.layout, 2, 0)
    rowsize!(fig.layout, 4, Relative(1 / 4))

    # # limits!(ax, 0, 10, 0, 10)

    display(fig)
end