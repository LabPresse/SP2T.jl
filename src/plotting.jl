function visualize_data(v::Video, g::GroundTruth)
    fig = Figure()
    ax = [
        Axis(fig[1, 1]),
        Axis(fig[2, 1]),
        Axis(fig[3, 1]),
        Axis(fig[1:4, 2][1, 1], aspect = DataAspect()),
        Axis(fig[1:4, 2][2, 1], aspect = DataAspect()),
    ]

    for i = 1:g.particle_num, j = 1:3
        lines!(ax[j], g.times, view(g.tracks, j, i, :))
    end

    sl_x = Slider(fig[4, 1], range = 1:v.params.length, startvalue = 1)

    frame1 = lift(sl_x.value) do x
        view(g.emitterPSF, :, :, x)
    end

    frame2 = lift(sl_x.value) do x
        view(v.data, :, :, x)
    end

    # for i = 1:3
    #     vspan!(ax[i], low, high)
    # end


    heatmap!(ax[4], frame1, colormap = :bone)
    heatmap!(ax[5], frame2, colormap = :bone)

    hidedecorations!.(ax[1:2])
    hidedecorations!.(ax[4:5])

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

function visualize_data_3D(v::Video, g::GroundTruth)
    fig = Figure()
    ax = [
        Axis(fig[1, 1]),
        Axis(fig[2, 1]),
        Axis(fig[3, 1]),
        Axis(fig[1:4, 2][1, 1], aspect = DataAspect()),
        Axis(fig[1:4, 2][2, 1], aspect = DataAspect()),
    ]

    for i = 1:g.particle_num, j = 1:3
        lines!(ax[j], g.times, view(g.tracks, j, i, :))
    end

    sl_x = Slider(fig[4, 1], range = 1:v.params.length, startvalue = 1)

    frame1 = lift(sl_x.value) do x
        view(g.emitterPSF, :, :, x)
    end

    frame2 = lift(sl_x.value) do x
        view(v.data, :, :, x)
    end

    # for i = 1:3
    #     vspan!(ax[i], low, high)
    # end


    heatmap!(ax[4], frame1, colormap = :bone)
    heatmap!(ax[5], frame2, colormap = :bone)

    hidedecorations!.(ax[1:2])
    hidedecorations!.(ax[4:5])

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
