function simulate(;
    diffusivity::Real,
    brightness::Real,
    nemitters::Integer,
    nframes::Integer,
    params::ExperimentalParameters,
    μ = nothing,
    σ = [0, 0, 0],
)
    T = typeof(params.period)
    x = Array{T}(undef, 3, nemitters, nframes)
    D = convert(T, diffusivity) * params.period
    h = convert(T, brightness) * params.period
    isnothing(μ) && (μ = framecenter(params))
    simulate!(x, μ, σ, D)
    return Sample(x, D, h, 0, one(T), zero(T), zero(T))
end

simulate(sample::Sample, params::ExperimentalParameters) =
    simframes(pxcounts(sample.tracks, sample.brightness, params))