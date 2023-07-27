abstract type AbstractPSF end

struct CircularGaussianLorenzian <: AbstractPSF
    z_ref::Float64 # [length] std of psf along z  (optical axis)
    σ_ref::Float64 # [length] std of psf along xy (image plane)
    σ_ref_sqrt2::Float64 # [length] std of psf along xy (image plane)
    function CircularGaussianLorenzian(na::Float64, n::Float64, λ::Float64)
        cosalphasqrt = sqrt(cos(asin(na / n)))
        cos32alpha = cosalphasqrt^3
        cos72alpha = cosalphasqrt^7
        a = λ / pi / n
        b = ((7 * (1 - cos32alpha)) / (4 - 7 * cos32alpha + 3 * cos72alpha))
        z_ref = a * b
        σ_ref = sqrt(a * z_ref) / 2
        new(z_ref, σ_ref, sqrt(2) * σ_ref)
    end
end

struct ExperimentalParameters
    units::Tuple{String,String}
    period::Float64
    exposure::Float64
    pxboundsx::Vector{Float64}
    pxboundsy::Vector{Float64}
    pxnumx::Int
    pxnumy::Int
    pxsize::Float64
    pxarea::Float64
    pxareatimesexposure::Float64
    NA::Float64
    nᵣ::Float64
    wavelength::Float64
    PSF::AbstractPSF
    validpx::Matrix{Bool}
    ExperimentalParameters(;
        units::Tuple{String,String,String} = ("μm", "s"),
        period::Float64 = 0.0033,
        exposure::Float64 = 0.003,
        validpx::AbstractMatrix{Bool} = ones(Bool, 50, 50),
        pxsize::Real = 0.133,
        NA::Float64 = 1.45,
        nᵣ::Float64 = 1.515,
        wavelength::Float64 = 0.665,
        PSF::AbstractPSF = CircularGaussianLorenzian(1.45, 1.515, 0.665),
        offsetx::Float64 = 0.0,
        offsety::Float64 = 0.0,
    ) = new(
        units,
        period,
        exposure,
        collect(range(offsetx, step = pxsize, length = size(validpx, 1) + 1)),
        collect(range(offsety, step = pxsize, length = size(validpx, 2) + 1)),
        size(validpx, 1),
        size(validpx, 2),
        pxsize,
        pxsize^2,
        pxsize^2 * exposure,
        NA,
        nᵣ,
        wavelength,
        PSF,
    )
end

# struct Priors
#     s::Categorical
#     x::MvNormal
#     D::InverseGamma
#     h::Gamma
#     F::Gamma
#     b::Bernoulli
#     Priors(;
#         p_s::AbstractVector{<:Real} = [0.5, 0.5],
#         μₓ::AbstractVector{<:Real} = [0, 0, 0],
#         σₓ::AbstractVector{<:Real} = [0, 0, 0],
#         # bleachrate_ϕ::Real = 1,
#         # bleachrate_ψ::Real = 1,
#         ϕ_D::Real = 1,
#         χ_D::Real = 1,
#         ϕ_F::Real = 1,
#         ψ_F::Real = 1,
#         ϕ_h::Real = 1,
#         ψ_h::Real = 1,
#         p_b::Real = 0.1,
#     ) = new(
#         Categorical(p_s),
#         MvNormal(μₓ, Diagonal(σₓ)),
#         # Gamma(bleachrate_ϕ, bleachrate_ψ / bleachrate_ϕ),
#         InverseGamma(ϕ_D, ϕ_D * χ_D),
#         Gamma(ϕ_h, ψ_h / ϕ_h),
#         Gamma(ϕ_F, ψ_F / ϕ_F),
#         Bernoulli(p_b),
#     )
# end

struct Priors
    x::MvNormal
    D::InverseGamma
    h::Gamma
    F::Gamma
    b::Bernoulli
    Priors(;
        μₓ = [0, 0, 0],
        σₓ = [0, 0, 0],
        ϕ_D = 1,
        χ_D = 1,
        ϕ_F = 1,
        ψ_F = 1,
        ϕ_h = 1,
        ψ_h = 1,
        p_b = 0.1,
    ) = new(
        MvNormal(μₓ, Diagonal(σₓ)),
        InverseGamma(ϕ_D, ϕ_D * χ_D),
        Gamma(ϕ_h, ψ_h / ϕ_h),
        Gamma(ϕ_F, ψ_F / ϕ_F),
        Bernoulli(p_b),
    )
end

struct Video
    data::BitArray{3}
    params::ExperimentalParameters
end

# mutable struct Sample{Tf<:AbstractFloat,Ti<:Integer}
#     b::Vector{Bool}
#     s::Array{Ti,3}
#     x::Array{Tf,3}
#     D::Tf
#     h::Tf
#     F::Matrix{Tf}
#     i::Int # iteration
#     T::Tf # temperature
#     logℙ::Tf # log posterior
# end


mutable struct Sample{Tf<:AbstractFloat}
    b::Vector{Bool}
    x::Array{Tf,3}
    D::Tf
    h::Tf
    F::Matrix{Tf}
    i::Int # iteration
    T::Tf # temperature
    logℙ::Tf # log posterior
    Sample(T) = new{T}()
    Sample(x, D, h, F; i = 0, T = 1.0, logℙ = NaN) =
        new{eltype(x)}(ones(Bool, size(x, 2)), x, D, h, F, i, T, logℙ)
end

# struct GroundTruth{Tf<:AbstractFloat,Ti<:Integer}
#     B::Int
#     x::Array{Tf,3}
#     s::Array{Ti,3}
#     D::Vector{Tf}
#     h::Tf
#     F::Tf
#     emitterPSF::Array{Tf,3}
# end

mutable struct Acceptances
    x::Int
    Acceptances() = new(0)
end

abstract type Annealing end

struct PolynomialAnnealing <: Annealing
    init_temperature::Real
    cutoff_iteration::Real
    order::Real
    PolynomialAnnealing(init_temperature, cutoff_iteration, order = 2) =
        new(init_temperature, cutoff_iteration, order)
end

get_temperature(i, a::PolynomialAnnealing)::Float64 =
    i >= a.cutoff_iteration || a.init_temperature * (i / a.cutoff_iteration - 1)^a.order

mutable struct Chain{Tf<:AbstractFloat}
    status::Sample{Tf}
    samples::Vector{Sample{Tf}}
    acceptances::Acceptances
    priors::Priors
    stride::Int
    size::Int
    sizelimit::Int
    Chain(T) = new{T}()
end