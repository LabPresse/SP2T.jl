# get_lowerlimit(px_min::Real, x::AbstractMatrix{<:Real}) = min(px_min, minimum(x))
# get_upperlimit(px_max::Real, x::AbstractMatrix{<:Real}) = max(px_max, maximum(x))

get_limits(px::AbstractVector{<:Real}, x::AbstractMatrix{<:Real}) =
    (min(px[1], minimum(x)), max(px[end], maximum(x)))

function visualize_data(v::Video, gt::GroundTruth)
    fig = Figure()
    ax = [
        Axis(fig[1, 1]),
        Axis(fig[2, 1]),
        Axis(fig[3, 1]),
        Axis(fig[1:4, 2][1, 1], aspect = DataAspect()),
        Axis(fig[1:4, 2][2, 1], aspect = DataAspect()),
    ]

    for i = 1:gt.particle_num, j = 1:3
        lines!(ax[j], gt.times, view(gt.tracks, j, i, :))
    end

    sl_x = Slider(fig[4, 1], range = 1:v.params.length, startvalue = 1)

    frame1 = lift(sl_x.value) do x
        view(gt.emitterPSF, :, :, x)
    end

    frame2 = lift(sl_x.value) do x
        view(v.data, :, :, x)
    end

    # for i = 1:3
    #     vspan!(ax[i], low, high)
    # end


    heatmap!(ax[4], frame1, colormap = :grays)
    heatmap!(ax[5], frame2, colormap = :grays)

    hidedecorations!.(ax[1:2])
    hidedecorations!.(ax[4:5])

    linkxaxes!(ax[1], ax[2], ax[3])

    xlims!(ax[3], 0, v.params.period * v.params.length)

    uls = (
        1.1 * max(
            maximum(view(gt.tracks, 1, :, :)),
            v.params.pixelboundsx[end] - v.params.pixelboundsx[1],
        ),
        1.1 * max(
            maximum(view(gt.tracks, 2, :, :)),
            v.params.pixelboundsy[end] - v.params.pixelboundsy[1],
        ),
        1.1 * maximum(view(gt.tracks, 3, :, :)),
    )
    lls = (
        1.1 * min(minimum(view(gt.tracks, 1, :, :)), 0),
        1.1 * min(minimum(view(gt.tracks, 2, :, :)), 0),
        1.1 * minimum(view(gt.tracks, 3, :, :)),
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

function visualize_data_3D(v::Video, gt::GroundTruth)
    fig = Figure()
    ax = [
        Axis3(fig[1:3, 1], zlabel = "t"),
        Axis(fig[4, 1], xlabel = "t", ylabel = "z"),
        Axis(fig[1:4, 2], aspect = DataAspect()),
        # Axis(fig[3:4, 2], aspect = DataAspect()),
    ]

    for m = 1:gt.particle_num
        lines!(ax[1], view(gt.tracks, 1, m, :), view(gt.tracks, 2, m, :), gt.times)
        lines!(ax[2], gt.times, view(gt.tracks, 3, m, :))
    end

    sl_x = Slider(fig[5, 1], range = 1:v.params.length, startvalue = 1)

    frame1 = lift(sl_x.value) do x
        view(gt.emitterPSF, :, :, x)
    end

    frame2 = lift(sl_x.value) do x
        view(v.data, :, :, x)
    end

    # for i = 1:3
    #     vlines!(ax[2], low, high)
    # end

    hm = heatmap!(
        ax[1],
        v.params.pixelboundsx,
        v.params.pixelboundsy,
        frame1,
        colormap = (:grays, 0.7),
    )

    vl = vlines!(ax[2], gt.times[1])

    on(sl_x.value) do n
        translate!(hm, 0, 0, gt.times[n])
        translate!(vl, gt.times[n], 0, -0.1)
    end

    heatmap!(
        ax[3],
        v.params.pixelboundsx,
        v.params.pixelboundsy .+ (2 * v.params.pixelsize + v.params.pixelboundsy[end]),
        frame1,
        colormap = :grays,
    )
    heatmap!(
        ax[3],
        v.params.pixelboundsx,
        v.params.pixelboundsy,
        frame2,
        colormap = :grays,
        colorrange = (false, true),
    )

    hidexdecorations!(ax[1], label = false)
    hideydecorations!(ax[1], label = false)
    hidedecorations!(ax[3])
    # hidedecorations!(ax[3])
    hidespines!(ax[3])

    # linkxaxes!(ax[1], ax[2], ax[3])

    # limits!(
    #     ax[1],
    #     v.params.pixelboundsx[1], v.params.pixelboundsx[end],
    #     v.params.pixelboundsy[1], v.params.pixelboundsy[end],
    # )

    (lowerx, upperx) = get_limits(v.params.pixelboundsx, view(gt.tracks, 1, :, :))
    (lowery, uppery) = get_limits(v.params.pixelboundsy, view(gt.tracks, 2, :, :))
    # limits!(ax[1], lowerx, upperx, lowery, uppery)

    xlims!(ax[1], lowerx, upperx)
    ylims!(ax[1], lowery, uppery)
    zlims!(ax[1], gt.times[1], gt.times[end])

    xlims!(ax[2], 0, gt.times[end])

    # uls = (
    #     1.1 * max(
    #         maximum(view(gt.tracks, 1, :, :)),
    #         v.params.pixelboundsx[end] - v.params.pixelboundsx[1],
    #     ),
    #     1.1 * max(
    #         maximum(view(gt.tracks, 2, :, :)),
    #         v.params.pixelboundsy[end] - v.params.pixelboundsy[1],
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
    #         uls .- (v.params.pixelnumx, v.params.pixelnumy, 0) .* (v.params.pixelsize / 2),
    #         (v.params.pixelnumx, v.params.pixelnumy, 0) .* (v.params.pixelsize / 2) .- lls,
    #     )
    # ylims!(
    #     ax[1],
    #     (v.params.pixelboundsx[end] - v.params.pixelboundsx[1]) / 2 - d[1],
    #     (v.params.pixelboundsx[end] - v.params.pixelboundsx[1]) / 2 + d[1],
    # )
    # ylims!(
    #     ax[2],
    #     (v.params.pixelboundsy[end] - v.params.pixelboundsy[1]) / 2 - d[2],
    #     (v.params.pixelboundsy[end] - v.params.pixelboundsy[1]) / 2 + d[2],
    # )
    # ylims!(ax[3], -d[3], d[3])

    colgap!(fig.layout, 1, 10)
    colsize!(fig.layout, 2, Relative(1 / 3))
    rowgap!(fig.layout, 1, 0)
    rowgap!(fig.layout, 2, 0)
    rowsize!(fig.layout, 4, Relative(1 / 4))

    # # limits!(ax, 0, 10, 0, 10)

    return fig
end