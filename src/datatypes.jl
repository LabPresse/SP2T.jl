IntegerOrNothing = Union{Integer,Nothing}

abstract type AbstractPSF end

struct CircularGaussianLorenzian{FT<:AbstractFloat} <: AbstractPSF
    z_ref::FT # [length] std of psf along z  (optical axis)
    σ_ref::FT # [length] std of psf along xy (image plane)
    σ_ref_sqrt2::FT # [length] std of psf along xy (image plane)
    function CircularGaussianLorenzian{FT}(NA::FT, nᵣ::FT, λ::FT) where {FT<:AbstractFloat}
        cos12α = sqrt(cos(asin(NA / nᵣ)))
        cos32α = cos12α^3
        cos72α = cos12α^7
        a = λ / pi / nᵣ
        b = ((7 * (1 - cos32α)) / (4 - 7 * cos32α + 3 * cos72α))
        z_ref = a * b
        σ_ref = sqrt(a * z_ref) / 2
        new{FT}(z_ref, σ_ref, sqrt(2) * σ_ref)
    end
    CircularGaussianLorenzian(
        z_ref::FT,
        σ_ref::FT,
        σ_ref_sqrt2::FT,
    ) where {FT<:AbstractFloat} = CircularGaussianLorenzian{FT}(z_ref, σ_ref, σ_ref_sqrt2)
end

struct ExperimentalParameters{FT<:AbstractFloat}
    units::Tuple{String,String}
    length::Int
    period::FT
    exposure::FT
    pxboundsx::AbstractVector{FT}
    pxboundsy::AbstractVector{FT}
    pxnumx::Int
    pxnumy::Int
    pxsize::FT
    pxarea::FT
    pxareatimesexposure::FT
    NA::FT
    nᵣ::FT
    λ::FT
    PSF::AbstractPSF
    ExperimentalParameters(
        FT::DataType;
        units::Tuple{String,String} = ("μm", "s"),
        length::Integer,
        period::Real,
        exposure::Real,
        pxnumx::Integer,
        pxnumy::Integer,
        pxsize::Real,
        NA::Real,
        nᵣ::Real,
        λ::Real,
        offsetx::Real = 0,
        offsety::Real = 0,
    ) = new{FT}(
        units,
        length,
        period,
        exposure,
        collect(range(offsetx, step = pxsize, length = pxnumx + 1)),
        collect(range(offsety, step = pxsize, length = pxnumy + 1)),
        pxnumx,
        pxnumy,
        pxsize,
        pxsize^2,
        pxsize^2 * exposure,
        NA,
        nᵣ,
        λ,
        CircularGaussianLorenzian(NA, nᵣ, λ),
    )
end
convert(FT::AbstractFloat, params::ExperimentalParameters) = ExperimentalParameters{FT}(
    params.units,
    params.length,
    params.period,
    params.exposure,
    params.pxboundsx,
    params.pxboundsy,
    params.pxnumx,
    params.pxnumy,
    params.pxsize,
    params.pxarea,
    params.pxareatimesexposure,
    params.NA,
    params.nᵣ,
    params.λ,
    params.PSF,
)

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
Priors(params::ExperimentalParameters) = Priors(
    μₓ = [params.pxnumx * params.pxsize / 2, params.pxnumy * params.pxsize / 2, 0],
    σₓ = [params.pxsize * 2, params.pxsize * 2, 0],
)

struct Video
    data::BitArray{3}
    params::ExperimentalParameters
    Video(p::ExperimentalParameters) = new(zeros(Bool, p.pxnumx, p.pxnumy, p.length), p)
end

mutable struct Sample{FT<:AbstractFloat}
    x::AbstractArray{FT,3}
    D::FT
    h::FT
    F::AbstractMatrix{FT}
    i::Int # iteration
    T::FT # temperature
    logℙ::FT # log posterior
    Sample(FT) = new{FT}()
    Sample(
        x::AbstractArray{FT,3},
        D::FT,
        h::FT,
        F::FT;
        i::Int = 0,
        T::FT = 1.0,
        logℙ::FT = NaN,
    ) where {FT<:AbstractFloat} = new{FT}(x, D, h, F, i, T, logℙ)
end
get_B(s::Sample) = size(s.x, 2)
get_type(s::Sample{FT}) where {FT} = FT

mutable struct Acceptances
    x::Int
    Acceptances() = new(0)
end

abstract type Annealing end

struct PolynomialAnnealing{FT<:AbstractFloat} <: Annealing
    init_temperature::FT
    cutoff_iteration::FT
    order::Int
    PolynomialAnnealing(
        init_temperature::FT,
        cutoff_iteration::FT,
        order::Int = 2,
    ) where {FT<:AbstractFloat} = new{FT}(init_temperature, cutoff_iteration, order)
end

get_temperature(i, a::PolynomialAnnealing)::Float64 =
    i >= a.cutoff_iteration || a.init_temperature * (i / a.cutoff_iteration - 1)^a.order

# FullSample contains auxiliary variables
mutable struct FullSample{FT<:AbstractFloat}
    b::BitVector
    x::AbstractArray{FT,3}
    D::FT
    h::FT
    F::AbstractMatrix{FT}
    i::Int # iteration
    T::FT # temperature
    logℙ::FT # log posterior
    FullSample(FT) = new{FT}()
    FullSample(
        b::BitVector,
        x::AbstractArray{FT,3},
        D::FT,
        h::FT,
        F::AbstractMatrix{FT};
        i::Int = 0,
        T::FT = 1.0,
        logℙ::FT = NaN,
    ) where {FT<:AbstractFloat} = new{FT}(b, x, D, h, F, i, T, logℙ)
end
get_B(s::Sample) = size(s.x, 2)
get_type(s::Sample{FT}) where {FT} = FT

mutable struct Chain{FT<:AbstractFloat}
    status::Sample{FT}
    samples::Vector{Sample{FT}}
    acceptances::Acceptances
    priors::Priors
    stride::Int
    size::Int
    sizelimit::Int
    Chain(T) = new{T}()
end