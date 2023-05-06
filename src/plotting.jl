function visualize(v::Video, g::GroundTruth)
    fig = Figure()
    ax = [
        Axis(fig[1, 1]),
        Axis(fig[2, 1]),
        Axis(fig[3, 1]),
        Axis(fig[1, 2]),
        Axis(fig[2, 2]),
        Axis(fig[3, 2]),
        Axis(fig[1:4, 3][1, 1], aspect = DataAspect()),
        Axis(fig[1:4, 3][2, 1], aspect = DataAspect()),
    ]

    for i = 1:g.particle_num, j = 1:3
        lines!(ax[j], g.times, view(g.tracks, j, i, :))
        lines!(ax[j+3], g.times, view(g.tracks, j, i, :))
    end

    xlims!.(ax[4:6], g.times[1], g.times[g.length_per_exposure])

    sl_x = Slider(fig[4, 1], range = 1:v.params.length, startvalue = 1)

    frame1 = lift(sl_x.value) do x
        view(g.emitterPSF, :, :, x)
    end

    frame2 = lift(sl_x.value) do x
        view(v.data, :, :, x)
    end

    low = lift(sl_x.value) do x
        g.times[g.length_per_exposure*(x-1)+1]
    end

    high = lift(sl_x.value) do x
        g.times[g.length_per_exposure*x]
    end

    for i = 1:3
        vspan!(ax[i], low, high)
    end

    onany(low, high) do l, h
        xlims!.(ax[4:6], l, h)
    end

    heatmap!(ax[7], frame1, colormap = :bone)
    heatmap!(ax[8], frame2, colormap = :bone)

    hidedecorations!.(ax[1:2])
    hidexdecorations!.(ax[3:3:6], ticks = false, ticklabels = false)
    hideydecorations!.(ax[3:3:6])
    hidedecorations!.(ax[4:5])
    hidedecorations!.(ax[7:8])

    linkxaxes!(ax[1], ax[2], ax[3])

    xlims!(ax[3], 0, v.params.period * v.params.length)

    uls = (
        1.1 * max(
            maximum(view(g.tracks, 1, :, :)),
            v.params.pixelboundsx[end] - v.params.pixelboundsx[1],
        ),
        1.1 * max(
            maximum(view(g.tracks, 2, :, :)),
            v.params.pixelboundsy[end] - v.params.pixelboundsy[1],
        ),
        1.1 * maximum(view(g.tracks, 3, :, :)),
    )
    lls = (
        1.1 * min(minimum(view(g.tracks, 1, :, :)), 0),
        1.1 * min(minimum(view(g.tracks, 2, :, :)), 0),
        1.1 * minimum(view(g.tracks, 3, :, :)),
    )

    d =
        max.(
            uls .- (v.params.pixelnumx, v.params.pixelnumy, 0) .* (v.params.pixelsize / 2),
            (v.params.pixelnumx, v.params.pixelnumy, 0) .* (v.params.pixelsize / 2) .- lls,
        )
    ylims!(
        ax[1],
        (v.params.pixelboundsx[end] - v.params.pixelboundsx[1]) / 2 - d[1],
        (v.params.pixelboundsx[end] - v.params.pixelboundsx[1]) / 2 + d[1],
    )
    ylims!(
        ax[2],
        (v.params.pixelboundsy[end] - v.params.pixelboundsy[1]) / 2 - d[2],
        (v.params.pixelboundsy[end] - v.params.pixelboundsy[1]) / 2 + d[2],
    )
    ylims!(ax[3], -d[3], d[3])

    colgap!(fig.layout, 1, 10)
    colsize!(fig.layout, 2, Relative(1 / 5))
    rowgap!(fig.layout, 1, 0)
    rowgap!(fig.layout, 2, 0)

    # limits!(ax, 0, 10, 0, 10)

    return fig
end
