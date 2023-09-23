Prior(param::ExperimentalParameter{FT}) where {FT} = Prior{FT}(
    μₓ = [param.pxnumx * param.pxsize / 2, param.pxnumy * param.pxsize / 2, 0],
    σₓ = [param.pxsize * 2, param.pxsize * 2, 0],
)

Sample(s::ChainStatus{FT}) where {FT} =
    Sample{FT}(Array(s.x.value[:, s.b.value, :]), s.D.value, s.h.value, s.i, s.𝑇, s.ln𝒫)

# Sample(s::ChainStatus) = Sample(s.x[:, 1:get_B(s), :], s.D, s.h, s.F, s.i, s.T, s.ln𝒫)

function set_b(B::Integer, M::Integer, 𝒫::Distribution)
    b = BitVector(zeros(Bool, M))
    b[1:B] .= true
    return DirectlySampledVectorRV(b, 𝒫)
end

function set_x(
    x::AbstractArray{FT,3},
    B::Integer,
    M::Integer,
    N::Integer,
    𝒫::Distribution,
    𝒬::Distribution,
) where {FT}
    newx = Array{FT,3}(undef, 3, M, N)
    newx[:, 1:B, :] = x
    return MetropolisHastingsVectorRV(newx, 𝒫, 𝒬)
end

set_D(D::Real, 𝒫::Distribution) = DirectlySampledScalarRV(D, 𝒫)

set_h(h::Real, 𝒫::Distribution, 𝒬::Distribution) = MetropolisHastingsScalarRV(h, 𝒫, 𝒬)

# load_prior = Bernoulli{FloatType}(0.5)
# init_pos_prior = default_init_pos_prior(param)
# diffusion_prior = InverseGamma{FloatType}(1, 1 * 1)
# emission_prior = Gamma{FloatType}(1, 1 / 1)

function ChainStatus(
    s::Sample{FT},
    M::Integer,
    exp_param::ExperimentalParameter{FT},
    prior_param::PriorParameter{FT},
) where {FT<:AbstractFloat}
    (~, B, N) = size(s.x)
    b = set_b(B, M, Bernoulli(prior_param.pb))
    x = set_x(
        s.x,
        B,
        M,
        N,
        MvNormal(prior_param.μx, prior_param.σx),
        MvNormal([exp_param.PSF.σ_ref, exp_param.PSF.σ_ref, exp_param.PSF.z_ref] ./ 2),
    )
    D = set_D(s.D, InverseGamma(prior_param.ϕD, prior_param.ϕD * prior_param.χD))
    h = set_h(s.h, Gamma(prior_param.ϕh, prior_param.ψh / prior_param.ϕh), Beta())
    G = get_pxPSF(s.x, exp_param.pxboundsx, exp_param.pxboundsy, exp_param.PSF)
    return ChainStatus(b, x, D, h, G, iszero(s.i) ? 1 : s.i, s.𝑇, s.ln𝒫)
    #TODO initialize 𝑇 and ln𝒫 better
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
    G = get_pxPSF(s.x, p.pxboundsx, p.pxboundsy, p.PSF)
    data = simulate_w(G, s.h, p.darkcounts, p.exposure)
    return Video(data, p)
end

function Chain(;
    initial_guess::Sample{FT},
    exp_param::ExperimentalParameter{FT},
    prior_param::PriorParameter{FT},
    max_emitter_num::Integer,
    sizelimit::Integer,
    annealing::Union{Annealing{FT},Nothing} = nothing,
) where {FT<:AbstractFloat}
    isnothing(annealing) && (annealing = PolynomialAnnealing{FT}())
    status = ChainStatus(initial_guess, max_emitter_num, exp_param, prior_param)
    return Chain{FT}(status, [initial_guess], annealing, 1, sizelimit)
end