get_limits(px::AbstractVector{<:Real}, x::AbstractMatrix{<:Real}) =
    (min(px[1], minimum(x)), max(px[end], maximum(x)))

function get_localization_error(S::AbstractVector, gt::Sample)
    X_gnd = view(gt.x, 1, :, :)
    Y_gnd = view(gt.x, 2, :, :)
    ~, B_gnd, N = size(gt.x)
    localization_errors = Vector{Float64}()
    order_matrix = Matrix{Int64}(undef, N, B_gnd)
    order_errors = Vector{Int64}()
    for s in S
        if size(s.x, 2) != B_gnd
            continue
        end
        total_error = 0
        X = view(s.x, 1, :, :)
        Y = view(s.x, 2, :, :)
        for n = 1:N
            min_error = Inf
            for (j, order) in enumerate(permutations(1:B_gnd, B_gnd))
                error = sum(
                    abs2.(view(X, order, n) .- view(X_gnd, :, n)) .+
                    abs2.(view(Y, order, n) .- view(Y_gnd, :, n)),
                )
                if min_error > error
                    min_error = error
                    order_matrix[n, :] .= order
                end
            end
            total_error += sqrt(min_error)
        end
        push!(localization_errors, total_error / N / B_gnd)
        push!(order_errors, count(!=(0), diff(order_matrix, dims = 1)))
    end
    return localization_errors, order_errors
end

function visualize(v::Video{FT}, gt::Sample{FT}) where {FT}
    if isa(v.frames, CuArray)
        to_cpu!(v)
    end
    data = v.frames
    p = v.param
    x = gt.x
    B = size(x, 2)

    g = get_px_PSF(gt.x, p.pxboundsx, p.pxboundsy, p.PSF)

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
        Text = (align = (:left, :top), font = "Arial", fontsize = 15),
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
    c::Chain;
    num_grid::Integer = 500,
    burn_in::Integer = 0,
) where {FT}
    if isa(v.frames, CuArray)
        to_cpu!(v)
    end
    histcolor =
        RGBAf(ColorSchemes.tab10[1].r, ColorSchemes.tab10[1].g, ColorSchemes.tab10[1].b, 1)
    set_theme!(my_theme())
    fig = Figure(; size = (800, 800), fontsize = 15, font = "Arial")
    ax = [
        Axis(fig[1, 1][1, 1], title = "x trajectory"),
        Axis(fig[2, 1][1, 1], xlabel = "Frame", title = "y trajectory"),
        Axis(fig[1, 2], xlabel = "Localization error (nm)"),
        Axis(fig[2, 2], xlabel = "Diffusion coefficient (μm²/s)"),
    ]

    s = @view c.samples[burn_in+1:end]
    all_trajectories = get_x(s)
    x = view(all_trajectories, :, :, 1)
    y = view(all_trajectories, :, :, 2)
    z = view(all_trajectories, :, :, 3)
    t = collect(1:v.param.length)
    @show ~, MAP_idx = findmax([i.ln𝒫 for i in s])

    B = size(x, 1)
    @show xrange = range(minimum(x), maximum(x), num_grid)
    @show yrange = range(minimum(y), maximum(y), num_grid)
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

    Colorbar(fig[0, 1], hm2, vertical = false, size = 10)
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
    ylims!(ax[1], 2.2, 4.9)
    ylims!(ax[2], 2.8, 4.2)
    for j = 1:size(s[MAP_idx].x, 2)
        lines!(
            ax[1],
            t,
            view(s[MAP_idx].x, 1, j, :);
            color = ColorSchemes.tab10[3],
            linewidth = 1.5,
        )
        lines!(
            ax[2],
            t,
            view(s[MAP_idx].x, 2, j, :);
            color = ColorSchemes.tab10[3],
            linewidth = 1.5,
        )
    end

    translate!(
        rangebars!(ax[1], [20], [4.9 - 0.25], [4.9], color = :black, linewidth = 3),
        0,
        0,
        -0.1,
    )
    translate!(text!(ax[1], 27, 4.9, text = "250 nm"), 0, 0, -0.1)

    translate!(
        rangebars!(ax[2], [20], [4.2 - 0.25], [4.2], color = :black, linewidth = 3),
        0,
        0,
        -0.1,
    )
    translate!(text!(ax[2], 27, 4.2, text = "250 nm"), 0, 0, -0.1)

    (localization_errors, order_errors) = get_localization_error(s, gt)
    hist!(
        ax[3],
        localization_errors .* 1000,
        color = ColorSchemes.tab10[10],
        normalization = :probability,
    )
    ylims!(ax[3], 0, nothing)

    D = get_D(s)
    D_CI = quantile(D, [0.025, 0.5, 0.975])
    vspan!(ax[4], D_CI[1], D_CI[3], color = :grey80)
    hist!(ax[4], D, color = histcolor, normalization = :pdf)
    @show D_range = range(minimum(D), maximum(D); length = 200)
    lines!(
        ax[4],
        D_range,
        pdf.(c.status.D.𝒫, D_range),
        color = ColorSchemes.tab10[4],
        linewidth = 3,
    )
    vlines!(ax[4], gt.D, color = ColorSchemes.tab10[2], linewidth = 3)
    xlims!(ax[4], 0.04, 0.12)
    ylims!(ax[4], 0, nothing)

    # B_range = range(minimum(result.B), maximum(result.B); length = 200)
    # B_prior = Binomial(100, 0.05)
    # # lines!(ax[4], B_range, pdf.(h_prior, h_range), color = ColorSchemes.tab10[4], linewidth = 1)
    # vlines!(ax[4], 3, color = ColorSchemes.tab10[2], linewidth = 1)
    # vlines!(ax[4], result.h_MAP, color = ColorSchemes.tab10[3], linewidth = 1)

    hideydecorations!.(ax[1:2])
    elem_1 =
        [LineElement(color = ColorSchemes.tab10[2], linestyle = nothing, linewidth = 2)]
    elem_2 =
        [LineElement(color = ColorSchemes.tab10[3], linestyle = nothing, linewidth = 2)]
    elem_3 =
        [LineElement(color = ColorSchemes.tab10[4], linestyle = nothing, linewidth = 2)]
    elem_5 = [PolyElement(color = :grey80, strokevisible = false)]

    Legend(
        fig[0, 2],
        [elem_1, elem_2, elem_3, elem_5],
        ["Ground truth", "MAP", "Prior", "95% CI"],
        framevisible = false,
        patchsize = (15, 15),
        nbanks = 2,
    )

    colsize!(fig.layout, 1, Relative(3 / 5))
    colsize!(fig.layout, 2, Relative(2 / 5))
    rowsize!(fig.layout, 0, Relative(1 / 10))
    display(fig)
    set_theme!()
end

