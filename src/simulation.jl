function simulate(;
    diffusivity::Real,
    brightness::Real,
    nemitters::Integer,
    nframes::Integer,
    data::Data,
    μ = nothing,
    σ = [0, 0, 0],
)
    T = typeof(data.period)
    x = Array{T}(undef, 3, nemitters, nframes)
    D = convert(T, diffusivity) * data.period
    h = convert(T, brightness) * data.period
    isnothing(μ) && (μ = framecenter(data))
    simulate!(x, μ, σ, D)
    return Sample(x, D, h, 0, one(T), zero(T), zero(T))
end

simulate!(data::Data, sample::Sample) =
    simframes!(data.frames, pxcounts(sample.tracks, sample.brightness, data))