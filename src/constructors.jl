Prior(param::ExperimentalParameter{FT}) where {FT} = Prior{FT}(
    μₓ = [param.pxnumx * param.pxsize / 2, param.pxnumy * param.pxsize / 2, 0],
    σₓ = [param.pxsize * 2, param.pxsize * 2, 0],
)

Sample(s::ChainStatus{FT}) where {FT} = Sample{FT}(
    Array(s.x.value[:, 1:s.M.value, :]),
    s.D.value,
    s.h.value,
    s.i,
    s.𝑇,
    s.ln𝒫,
    s.lnℒ,
)

# Sample(s::ChainStatus) = Sample(s.x[:, 1:get_B(s), :], s.D, s.h, s.F, s.i, s.T, s.ln𝒫)

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

# load_prior = Bernoulli{FloatType}(0.5)
# init_pos_prior = default_init_pos_prior(param)
# diffusion_prior = InverseGamma{FloatType}(1, 1 * 1)
# emission_prior = Gamma{FloatType}(1, 1 / 1)

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
        MvNormal([exp_param.PSF.σ_ref, exp_param.PSF.σ_ref, exp_param.PSF.z_ref] ./ 2),
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

# function Sample(
#     FloatType::DataType;
#     param::ExperimentalParameter,
#     prior::Prior,
#     emitter_number::Integer,
#     diffusion_coefficient::Real,
#     emission_rate::Real,
# )
#     ftypeof(param) ≡ FloatType ||
#         @warn "The float type in the argument is different from that of the experimental parameter!"
#     B, N, T, D, h, F = emitter_number,
#     param.length,
#     FloatType(param.period),
#     FloatType(diffusion_coefficient),
#     FloatType(emission_rate),
#     param.darkcounts

#     x = Array{FloatType,3}(undef, 3, B, N)
#     simulate!(x, prior.x, D, T)
#     return Sample(x, D, h, F)
# end

function Video(p::ExperimentalParameter, s::Sample)
    ftypeof(p) ≡ ftypeof(s) ||
        @warn "Float type mismatch between the experimental parameter and the sample!"
    𝐔 = get_px_intensity(s.x, p.pxboundsx, p.pxboundsy, s.h * p.period, p.darkcounts, p.PSF)
    𝐖 = intensity2frame(𝐔)
    return Video(𝐖, p)
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
    update_ln𝒫!(status, video, CPU())
    chain = Chain{FT}(status, Sample{FT}[], annealing, 1, sizelimit)
    extend!(chain)
    return chain
end