using SP2T
using JLD2

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

ROIindices = getROIindices(indices, [291 191 270*255+1; 340 240 300*255], 512, 512);

frames = getframes(ROIindices, width = 50, height = 50, batchsize = 1);

darkcounts = get_darkcounts(frames)

idx = darkcounts .== 0
@views darkcounts[idx] .+= eps()

jldsave("./data/beads/darkcounts1.jld2"; darkcounts)
