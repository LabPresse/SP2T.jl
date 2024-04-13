Prior(param::ExperimentalParameter{FT}) where {FT} = Prior{FT}(
    μₓ = [param.pxboundsx[end], param.pxboundsy[end], 0] ./ 2,
    σₓ = getpxsize(param) .* [2, 2, 0],
)

Sample(s::ChainStatus{FT}) where {FT} = Sample{FT}(
    Array(s.tracks.value[:, 1:s.emittercount.value, :]),
    s.D.value,
    s.h.value,
    s.i,
    s.𝑇,
    s.ln𝒫,
    s.lnℒ,
)

function set_b(B::Integer, M::Integer, dynamics::Dynamics, 𝒫::Distribution)
    b = BitVector(zeros(Bool, M))
    b[1:B] .= true
    return DSTrajectory(b, dynamics, 𝒫)
end

function set_b(B::Integer, M::Integer, 𝒫::Distribution)
    b = BitVector(zeros(Bool, M))
    b[1:B] .= true
    return DSIID(b, 𝒫)
end

function set_x(
    x::AbstractArray{FT,3},
    B::Integer,
    M::Integer,
    N::Integer,
    dynamics::Dynamics,
    𝒫::Distribution,
    𝒬::Distribution,
) where {FT}
    newx = Array{FT,3}(undef, 3, M, N)
    newx[:, 1:B, :] = x
    return MHTrajectory(newx, dynamics, 𝒫, 𝒬)
end

set_M(M::Integer, 𝒫::Distribution) = DSIID(M, 𝒫)

set_D(D::Real, 𝒫::Distribution) = DSIID(D, 𝒫)

set_h(h::Real, 𝒫::Distribution, 𝒬::Distribution) = MHIID(h, 𝒫, 𝒬)

function ChainStatus(
    s::Sample{FT},
    ℳ::Integer,
    exp_param::ExperimentalParameter{FT},
    prior_param::PriorParameter{FT},
) where {FT<:AbstractFloat}
    (~, B, N) = size(s.x)
    x = set_x(
        s.x,
        B,
        ℳ,
        N,
        Brownian(),
        MvNormal(prior_param.μx, prior_param.σx),
        MvNormal([exp_param.PSF.σ₀, exp_param.PSF.σ₀, exp_param.PSF.z₀] ./ 2),
    )
    M = set_M(size(s.x, 2), Geometric(1 - prior_param.qM))
    D = set_D(s.D, InverseGamma(prior_param.ϕD, prior_param.ϕD * prior_param.χD))
    h = set_h(s.h, Gamma(prior_param.ϕh, prior_param.ψh / prior_param.ϕh), Beta())
    𝐔 = get_px_intensity(
        s.x,
        exp_param.pxboundsx,
        exp_param.pxboundsy,
        s.h * exp_param.period,
        exp_param.darkcounts,
        exp_param.PSF,
    )
    return ChainStatus(x, M, D, h, 𝐔, iszero(s.i) ? 1 : s.i, s.𝑇, s.ln𝒫, s.lnℒ)
    #TODO initialize 𝑇 better
end

function Video(p::ExperimentalParameter, s::Sample, meta::Dict{String,Any})
    _eltype(p) ≡ _eltype(s) ||
        @warn "Float type mismatch between the experimental parameter and the sample!"
    𝐔 = get_px_intensity(s.x, p.pxboundsx, p.pxboundsy, s.h * p.period, p.darkcounts, p.PSF)
    𝐖 = _getframes(𝐔)
    return Video(𝐖, p, meta)
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