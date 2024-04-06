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
    Poly = (strokecolor = ColorSchemes.tab10[1], strokewidth = 2),
)

function range_validator(s::String)
    parsed_vec = tryparse.(Int64, split(s, ','))
    return eltype(parsed_vec) == Int64 && length(parsed_vec) == 2
end

getrect(w_sl, h_sl) = @lift Rect(
    $(w_sl.interval)[1] - 0.5,
    $(h_sl.interval)[1] - 0.5,
    $(w_sl.interval)[2] - $(w_sl.interval)[1] + 1,
    $(h_sl.interval)[2] - $(h_sl.interval)[1] + 1,
)

getvertices(rect) = (@lift $(rect).origin), (@lift $(rect).origin .+ $(rect).widths)

function view_frames(frames::AbstractArray{<:Integer}, batchsize::Integer)
    framewidth, frameheight, framecount = size(frames)

    set_theme!(theme_data_viewer)
    fig = Figure(; size = (1000, 300))
    axes = [Axis(fig[1, 1]), Axis(fig[1, 3][1, 1]), Axis(fig[1, 3][2, 1])]

    h_sl = IntervalSlider(
        fig[1, 2],
        range = 1:frameheight,
        startvalues = (1, frameheight),
        horizontal = false,
    )
    w_sl = IntervalSlider(fig[2, 1], range = 1:framewidth, startvalues = (1, framewidth))
    tb = Textbox(
        fig[2, 3][1, 2],
        halign = :right,
        placeholder = "start frame, end frame",
        validator = range_validator,
        width = 150,
    )
    slgrid = SliderGrid(
        fig[3, :][1, 1],
        (
            label = "Frame",
            range = 1:framecount,
            startvalue = 1,
            snap = false,
            format = x -> "$x/$framecount",
        ),
        (
            label = "Contrast",
            range = 1:batchsize,
            startvalue = batchsize,
            snap = false,
            format = x -> "$x/$batchsize",
        ),
    )
    button = Button(fig[3, :][1, 2], label = "print")

    lowerindex = Observable(1)
    upperindex = Observable(framecount)

    on(tb.stored_string) do str
        interval = sort!(parse.(Int64, split(str, ',')))
        lowerindex[] = ifelse(1 <= interval[1] <= framecount, interval[1], lowerindex[])
        upperindex[] = ifelse(1 <= interval[2] <= framecount, interval[2], upperindex[])
    end

    sl_tags = (@lift "($($(w_sl.interval)[1]), $($(h_sl.interval)[1]))"),
    (@lift "($($(w_sl.interval)[2]), $($(h_sl.interval)[2]))")
    rangetag =
        @lift "Frame range: $(min($lowerindex,$upperindex))-$(max($lowerindex,$upperindex))"

    colorrange = @lift (0, $(slgrid.sliders[2].value))

    mainframe = @lift view(frames, :, :, $(slgrid.sliders[1].value))
    startframe = @lift view(frames, :, :, $lowerindex)
    endframe = @lift view(frames, :, :, $upperindex)

    on(button.clicks) do n
        println(
            "Bits summed:\t$batchsize\nROI:\t$(sl_tags[1][])--$(sl_tags[2][])\nFrame range:\t$(lowerindex[])--$(upperindex[])",
        )
    end

    heatmap!(
        axes[1],
        1:framewidth,
        1:frameheight,
        mainframe,
        colormap = :bone,
        colorrange = colorrange,
    )
    heatmap!(
        axes[2],
        1:framewidth,
        1:frameheight,
        startframe,
        colormap = :bone,
        colorrange = colorrange,
    )
    heatmap!(
        axes[3],
        1:framewidth,
        1:frameheight,
        endframe,
        colormap = :bone,
        colorrange = colorrange,
    )
    limits!.(axes, 0.5, framewidth + 0.5, 0.5, frameheight + 0.5)

    rect = getrect(w_sl, h_sl)
    for ax in axes
        poly!(ax, rect, color = (:white, 0))
    end

    verts = getvertices(rect)
    text!(axes[1], verts[1], align = (:left, :bottom), text = sl_tags[1], color = :white)
    text!(axes[1], verts[2], align = (:right, :top), text = sl_tags[2], color = :white)

    Label(fig[0, 1:2], "Frame viewer")
    Label(fig[0, 3], "Range viewer")
    Label(fig[2, 3][1, 1], rangetag, halign = :left)

    colsize!(fig.layout, 1, Relative(2 / 3))
    colsize!(fig.layout, 3, Relative(1 / 3))
    rowsize!(fig.layout, 1, Aspect(1, frameheight / framewidth))
    resize_to_layout!(fig)
    display(fig)
    set_theme!()
end