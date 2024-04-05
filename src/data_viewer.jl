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
    w, h, c = size(frames)

    set_theme!(theme_data_viewer)
    fig = Figure(; size = (1000, 300))
    axes = [Axis(fig[1, 1]), Axis(fig[1, 3][1, 1]), Axis(fig[1, 3][2, 1])]

    h_sl = IntervalSlider(fig[1, 2], range = 1:h, startvalues = (1, h), horizontal = false)
    w_sl = IntervalSlider(fig[2, 1], range = 1:w, startvalues = (1, w))
    range_tb = Textbox(
        fig[2, 3][1, 2],
        halign = :right,
        placeholder = "start frame, end frame",
        validator = range_validator,
        width = 150,
    )
    f_sl = SliderGrid(
        fig[3, :],
        (
            label = "Current\nframe",
            range = 1:c,
            startvalue = 1,
            snap = false,
            format = x -> "$x/$c",
        ),
    )
    c_sl = SliderGrid(
        fig[4, :][1, 1],
        (
            label = "Contrast",
            range = 1:batchsize,
            startvalue = batchsize,
            snap = false,
            format = x -> "$x/$batchsize",
        ),
    )
    button = Button(fig[4, :][1, 2], label = "print")

    lowerindex = Observable(1)
    upperindex = Observable(c)

    on(range_tb.stored_string) do str
        interval = sort!(parse.(Int64, split(str, ',')))
        lowerindex[] = ifelse(1 <= interval[1] <= c, interval[1], lowerindex[])
        upperindex[] = ifelse(1 <= interval[2] <= c, interval[2], upperindex[])
    end

    sl_tags = (@lift "($($(w_sl.interval)[1]), $($(h_sl.interval)[1]))"),
    (@lift "($($(w_sl.interval)[2]), $($(h_sl.interval)[2]))")
    rangetag =
        @lift "Frame range: $(min($lowerindex,$upperindex))-$(max($lowerindex,$upperindex))"

    colorrange = @lift (0, $(c_sl.sliders[1].value))

    mainframe = @lift view(frames, :, :, $(f_sl.sliders[1].value))
    startframe = @lift view(frames, :, :, $lowerindex)
    endframe = @lift view(frames, :, :, $upperindex)

    on(button.clicks) do n
        println(
            "Bits summed:\t$batchsize\nROI:\t$(sl_tags[1][])--$(sl_tags[2][])\nFrame range:\t$(lowerindex[])--$(upperindex[])",
        )
    end

    @show c_sl.sliders[1].value

    heatmap!(axes[1], 1:w, 1:h, mainframe, colormap = :bone, colorrange = colorrange)
    heatmap!(axes[2], 1:w, 1:h, startframe, colormap = :bone, colorrange = colorrange)
    heatmap!(axes[3], 1:w, 1:h, endframe, colormap = :bone, colorrange = colorrange)
    limits!.(axes, 0.5, w + 0.5, 0.5, h + 0.5)

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
    rowsize!(fig.layout, 1, Aspect(1, h / w))
    resize_to_layout!(fig)
    display(fig)
    set_theme!()
end