# forward functions for spatial tracks
"""
    fill!(x, D, T)

Generate Brownian spatial trajectories `x` with diffusion coefficient `D` and
period `T`. Initial positions `x[:, :, 1]` are required. The three indices of
`x` are dimension, particle, and time, respectively. This convention is used to
optimize operations on a time point.
"""
function evolve!(x::AbstractArray{FT,3}, σ::FT) where {FT<:AbstractFloat}
    (~, M, N) = size(x)
    x[:, :, 2:N] = σ .* randn(eltype(x), 3, M, N - 1)
    cumsum!(x, x, dims = 3)
    return x
end

function simulate!(
    x::AbstractArray{FT,3},
    prior::Distribution,
    D::FT,
    τ::FT,
) where {FT<:AbstractFloat}
    x[:, :, 1] = rand(prior, size(x, 2))
    evolve!(x, sqrt(2 * D * τ))
    return x
end

# forward functions that calculate contributions from emitters
function get_erf(x, xᵖ, σ)
    U = (xᵖ .- x) ./ σ
    return @views erf.(U[1:end-1, :, :], U[2:end, :, :]) ./ 2
end

function simulate!(
    g::AbstractArray{FT,3},
    x::AbstractArray{FT,3},
    xᵖ::AbstractVector{FT},
    yᵖ::AbstractVector{FT},
    PSF::CircularGaussianLorenzian{FT},
) where {FT<:AbstractFloat}
    σ_sqrt2 = PSF.σ_ref_sqrt2 .* sqrt.(1 .+ (view(x, 3:3, :, :) ./ PSF.z_ref) .^ 2)
    U = get_erf(view(x, 1:1, :, :), xᵖ, σ_sqrt2)
    V = get_erf(view(x, 2:2, :, :), yᵖ, σ_sqrt2)
    return batched_mul!(g, U, batched_adjoint(V))
end

function simulate_G(
    x::AbstractArray{FT,3},
    xᵖ::AbstractVector{FT},
    yᵖ::AbstractVector{FT},
    PSF::CircularGaussianLorenzian{FT},
) where {FT<:AbstractFloat}
    σ_sqrt2 = PSF.σ_ref_sqrt2 .* sqrt.(1 .+ (view(x, 3:3, :, :) ./ PSF.z_ref) .^ 2)
    U = get_erf(view(x, 1:1, :, :), xᵖ, σ_sqrt2)
    V = get_erf(view(x, 2:2, :, :), yᵖ, σ_sqrt2)
    return batched_mul(U, batched_adjoint(V))
end

# sim_frames(u) = rand(size(u)) .< -expm1.(-u)

# g = sim_img(x, param.pxboundsx, param.pxboundsy, param.PSF)

function simulate!(
    v::BitArray{3},
    g::AbstractArray{FT,3},
    h::FT,
    F::AbstractMatrix{FT},
    τ::FT,
    τA::FT,
) where {FT<:AbstractFloat}
    u = @. F * τA + h * g * τ
    v .= rand(eltype(u), size(u)) .< -expm1.(-u)
    return v
end

function simulate_w(
    G::AbstractArray{FT,3},
    h::FT,
    F::AbstractMatrix{FT},
    τ::FT,
) where {FT<:AbstractFloat}
    u = @. F + h * G * τ
    return rand(eltype(u), size(u)) .< -expm1.(-u)
end

# function simulate(M::Integer, p::Real)
#     B = rand(Binomial(M, p))
#     b = zeros(Bool, M)
#     b[1:B] = true
#     return b
# end

# function simulate!(
#     s::AbstractSample;
#     param::ExperimentalParameter,
#     prior::Prior,
#     emitter_number::Integer,
#     diffusion_coefficient::Real,
#     emission_rate::Real,
#     background_flux::Matrix{<:Real},
# )
#     B, N, T = emitter_number, param.length, param.period
#     s.D, s.h, s.F = diffusion_coefficient, emission_rate, background_flux
#     s.x = Array{get_type(s),3}(undef, 3, B, N)
#     simulate!(s.x, prior.x, s.D, T)
#     return s
# end

# function simulate!(v::Video, s::Sample)
#     param = v.param
#     g = Array{ftypeof(s),3}(undef, param.pxnumx, param.pxnumy, param.length)
#     simulate!(g, s.x, param.pxboundsx, param.pxboundsy, param.PSF)
#     simulate!(v.data, g, s.h, s.F, param.exposure, param.pxareatimesexposure)
#     return v
# end

# function initialize!(s::FullSample; p::ExperimentalParameter, M::Integer, ℙ::Prior)
#     s.D = rand(ℙ.D)
#     s.h = rand(ℙ.h)
#     s.F = rand(ℙ.F, p.pxnumx, p.pxnumy)
#     simulate!(
#         s,
#         param = p,
#         prior = ℙ,
#         emitter_number = M,
#         diffusion_coefficient = s.D,
#         emission_rate = s.h,
#         background_flux = s.F,
#     )
#     return s
# end

function simulate_sample(;
    param::ExperimentalParameter{FT},
    emitter_number::Integer,
    diffusion_coefficient::Real,
    emission_rate::Real,
    init_pos_prior::Union{Missing,MultivariateDistribution} = missing,
) where {FT}
    B, N, T, D, h = emitter_number,
    param.length,
    param.period,
    FT(diffusion_coefficient),
    FT(emission_rate)
    if ismissing(init_pos_prior)
        init_pos_prior = default_init_pos_prior(param)
    end
    x = Array{FT,3}(undef, 3, B, N)
    simulate!(x, init_pos_prior, D, T)
    return Sample(x, D, h)
end