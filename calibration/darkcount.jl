using SP2T
using JLD2
using GLMakie
using ColorSchemes

function get_darkcounts(frames)
    darkcounts = Array{Float64}(undef, size(frames, 1), size(frames, 2))
    sum!(darkcounts, frames)
    N = size(frames, 3)
    @. darkcounts .= -log1p(-darkcounts / N)
    return darkcounts
end

indices = readbin(
    "/home/lancexwq/Dropbox (ASU)/SinglePhotonTracking/Data/Weiqing-Nathan/Diffusing Beads/GYbeads_SPADdirect_b2_10p3us/",
);

DCindices = extractROI(indices, (512, 512), (291, 131, 1), (340, 180, 30 * 255));

DCframes = getframes(DCindices, width = 50, height = 50, batchsize = 1);

darkcounts = get_darkcounts(DCframes)

idx = darkcounts .== 0
@views darkcounts[idx] .+= eps()

fig = Figure()
ax = Axis(fig[1, 1], aspect = DataAspect())
heatmap!(ax, darkcounts, colormap = :bone)
fig

jldsave("./data/beads/darkcounts2.jld2"; darkcounts)
