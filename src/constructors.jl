Prior(param::ExperimentalParameter{FT}) where {FT} = Prior{FT}(
    μₓ = [param.pxboundsx[end], param.pxboundsy[end], 0] ./ 2,
    σₓ = getpxsize(param) .* [2, 2, 0],
)

Sample(s::ChainStatus{FT}) where {FT} = Sample{FT}(
    Array(s.tracks.value[:, 1:s.emittercount.value, :]),
    s.diffusivity.value,
    s.brightness.value,
    s.iteration,
    s.temperature,
    s.logposterior,
    s.loglikelihood,
)

function set_b(
    emittercount::Integer,
    maxcount::Integer,
    dynamics::Dynamics,
    prior::Distribution,
)
    b = BitVector(zeros(Bool, maxcount))
    b[1:emittercount] .= true
    return DSTrajectory(b, dynamics, prior)
end

function set_b(emittercount::Integer, maxcount::Integer, prior::Distribution)
    b = BitVector(zeros(Bool, maxcount))
    b[1:emittercount] .= true
    return DSIID(b, prior)
end

function set_x(
    tracks::AbstractArray{FT,3},
    emittercount::Integer,
    maxcount::Integer,
    framecount::Integer,
    dynamics::Dynamics,
    prior::Distribution,
    proposal::Distribution,
) where {FT}
    newx = Array{FT,3}(undef, 3, maxcount, framecount)
    newx[:, 1:emittercount, :] = tracks
    return MHTrajectory(newx, dynamics, prior, proposal)
end

set_M(M::Integer, prior::Distribution) = DSIID(M, prior)

set_D(D::Real, prior::Distribution) = DSIID(D, prior)

set_h(h::Real, prior::Distribution, proposal::Distribution) = MHIID(h, prior, proposal)

function ChainStatus(
    sample::Sample{FT},
    maxcount::Integer,
    param::ExperimentalParameter{FT},
    prior_param::PriorParameter{FT},
) where {FT<:AbstractFloat}
    (~, B, N) = size(sample.tracks)
    tracks = set_x(
        sample.tracks,
        B,
        maxcount,
        N,
        Brownian(),
        MvNormal(prior_param.μx, prior_param.σx),
        MvNormal([param.PSF.σ₀, param.PSF.σ₀, param.PSF.z₀] ./ 2),
    )
    M = set_M(size(sample.tracks, 2), Geometric(1 - prior_param.qM))
    diffusivity = set_D(
        sample.diffusivity,
        InverseGamma(prior_param.ϕD, prior_param.ϕD * prior_param.χD),
    )
    brightness = set_h(
        sample.brightness,
        Gamma(prior_param.ϕh, prior_param.ψh / prior_param.ϕh),
        Beta(),
    )
    𝐔 = get_px_intensity(
        sample.tracks,
        param.pxboundsx,
        param.pxboundsy,
        sample.brightness * param.period,
        param.darkcounts,
        param.PSF,
    )
    return ChainStatus(
        tracks,
        M,
        diffusivity,
        brightness,
        𝐔,
        iszero(sample.iteration) ? 1 : sample.iteration,
        sample.temperature,
        sample.logposterior,
        sample.loglikelihood,
    )
    #TODO initialize 𝑇 better
end

function Video(param::ExperimentalParameter, groundtruth::Sample, meta::Dict{String,Any})
    _eltype(param) ≡ _eltype(groundtruth) ||
        @warn "Float type mismatch between the experimental parameter and the sample!"
    𝐔 = get_px_intensity(
        groundtruth.tracks,
        param.pxboundsx,
        param.pxboundsy,
        groundtruth.brightness * param.period,
        param.darkcounts,
        param.PSF,
    )
    frames = _getframes(𝐔)
    return Video(frames, param, meta)
end

function Chain(;
    initial_guess::Sample{FT},
    video::Video{FT},
    prior_param::PriorParameter{FT},
    max_emitter_num::Integer,
    sizelimit::Integer,
    annealing::Union{Annealing{FT},Nothing} = nothing,
) where {FT<:AbstractFloat}
    isnothing(annealing) && (annealing = PolynomialAnnealing{FT}())
    to_cpu!(video)
    status = ChainStatus(initial_guess, max_emitter_num, video.param, prior_param)
    update_off_x!(status, video.param, CPU())
    update_ln𝒫!(status, video, CPU())
    chain = Chain{FT}(status, Sample{FT}[], annealing, 1, sizelimit)
    extend!(chain)
    return chain
end