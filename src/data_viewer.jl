theme_data_viewer = Theme(
    Axis = (
        aspect = DataAspect(),
        bottomspinevisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        xgridvisible = false,
        xticklabelsvisible = false,
        xticksvisible = false,
        ygridvisible = false,
        yticklabelsvisible = false,
        yticksvisible = false,
    ),
)

function range_validator(s::String)
    parsed_vec = tryparse.(Int64, split(s, ','))
    return eltype(parsed_vec) == Int64 && length(parsed_vec) == 2
end

function view_frames(
    frames::AbstractArray{<:Integer};
    max_intensity::Union{Integer,Nothing} = nothing,
)
    isnothing(max_intensity) && (max_intensity = maximum(frames))
    framewidth, frameheight, framecount = size(frames)
    set_theme!(theme_data_viewer)

    fig = Figure(; size = (1000, 300))
    axes = [Axis(fig[1, 1]), Axis(fig[1, 3][1, 1]), Axis(fig[1, 3][2, 1])]

    widthslider =
        IntervalSlider(fig[2, 1], range = 1:framewidth, startvalues = (1, framewidth))
    heightslider = IntervalSlider(
        fig[1, 2],
        range = 1:frameheight,
        startvalues = (1, frameheight),
        horizontal = false,
    )
    frameslider = SliderGrid(
        fig[3, :],
        (
            label = "Current\nframe",
            range = 1:framecount,
            startvalue = 1,
            snap = false,
            format = x -> "$x/$framecount",
        ),
    )

    textbox = Textbox(
        fig[2, 3][1, 2],
        placeholder = "start frame, end frame",
        validator = range_validator,
    )
    startframeindex = Observable(1)
    endframeindex = Observable(framecount)

    on(textbox.stored_string) do str
        interval = sort!(parse.(Int64, split(str, ',')))
        startframeindex[] =
            ifelse(1 <= interval[1] <= framecount, interval[1], startframeindex[])
        endframeindex[] =
            ifelse(1 <= interval[2] <= framecount, interval[2], endframeindex[])
    end

    bottomlefttext = lift(widthslider.interval, heightslider.interval) do w_int, h_int
        w, h = w_int[1], h_int[1]
        "($w, $h)"
    end

    toprighttext = lift(widthslider.interval, heightslider.interval) do w_int, h_int
        w, h = w_int[2], h_int[2]
        "($w, $h)"
    end

    framerangetext = lift(startframeindex, endframeindex) do val1, val2
        minval, maxval = minmax(val1, val2)
        "Frame range: $minval-$maxval"
    end

    mainframe = lift(frameslider.sliders[1].value) do val
        view(frames, :, :, val)
    end

    startframe = lift(startframeindex) do val
        view(frames, :, :, val)
    end

    endframe = lift(endframeindex) do val
        view(frames, :, :, val)
    end

    # color points differently if they are within the two intervals

    # pointsx = rand(1:framewidth, 100)
    # pointsy = rand(1:frameheight, 100)

    # colors = lift(widthslider.interval, heightslider.interval) do h_int, v_int
    #     @. (h_int[1] <= pointsx <= h_int[2]) & (v_int[1] <= pointsy <= v_int[2])
    # end

    lbound = lift(widthslider.interval) do w_int
        w_int[1] - 0.5
    end
    rbound = lift(widthslider.interval) do w_int
        w_int[2] + 0.5
    end
    bbound = lift(heightslider.interval) do h_int
        h_int[1] - 0.5
    end
    tbound = lift(heightslider.interval) do h_int
        h_int[2] + 0.5
    end

    xmin = lift(widthslider.interval) do w_int
        (w_int[1] - 1) / framewidth
    end
    xmax = lift(widthslider.interval) do w_int
        w_int[2] / framewidth
    end
    ymin = lift(heightslider.interval) do h_int
        (h_int[1] - 1) / frameheight
    end
    ymax = lift(heightslider.interval) do h_int
        h_int[2] / frameheight
    end

    heatmap!(axes[1], 1:framewidth, 1:frameheight, mainframe, colormap = :bone)
    heatmap!(axes[2], 1:framewidth, 1:frameheight, startframe, colormap = :bone)
    heatmap!(axes[3], 1:framewidth, 1:frameheight, endframe, colormap = :bone)

    limits!.(axes[1:3], 0.5, framewidth + 0.5, 0.5, frameheight + 0.5)

    for ax in axes
        vlines!(ax, lbound, ymin = ymin, ymax = ymax, color = ColorSchemes.tab10[1])
        vlines!(ax, rbound, ymin = ymin, ymax = ymax, color = ColorSchemes.tab10[1])
        hlines!(ax, bbound, xmin = xmin, xmax = xmax, color = ColorSchemes.tab10[1])
        hlines!(ax, tbound, xmin = xmin, xmax = xmax, color = ColorSchemes.tab10[1])
    end

    text!(axes[1], lbound, bbound, align = (:left, :bottom), text = bottomlefttext)
    text!(axes[1], rbound, tbound, align = (:right, :top), text = toprighttext)

    Label(fig[0, 1:2], "Frame viewer")
    Label(fig[0, 3], "Range viewer")
    Label(fig[2, 3][1, 1], framerangetext, justification = :left)

    colsize!(fig.layout, 1, Relative(2 / 3))
    colsize!(fig.layout, 3, Relative(1 / 3))
    rowsize!(fig.layout, 1, Aspect(1, frameheight / framewidth))
    resize_to_layout!(fig)
    display(fig)
    set_theme!()
end