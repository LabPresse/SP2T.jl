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
    units::Tuple{String,String,String}
    length::Int
    period::Float64
    exposure::Float64
    pixelboundsx::Vector{Float64}
    pixelboundsy::Vector{Float64}
    pixelnumx::Int
    pixelnumy::Int
    pixelsize::Float64
    areatimesexposure::Float64
    na::Float64
    refractiveindex::Float64
    wavelength::Float64
    psf::AbstractPSF
    ExperimentalParameters(;
        length::Int=50,
        period::Float64=0.033,
        exposure::Float64=0.03,
        pixelnumx::Int=15,
        pixelnumy::Int=23,
        pixelsize::Real=0.133,
        na::Float64=1.45,
        refractiveindex::Float64=1.515,
        wavelength::Float64=0.665,
        psf::AbstractPSF=CircularGaussianLorenzian(1.45,1.515,0.665),
        units::Tuple{String,String,String} = ("μm", "s","ADU"),
        offsetx::Float64 = 0.0,
        offsety::Float64 = 0.0,
    ) = new(
        units,
        length,
        period,
        exposure,
        collect(range(offsetx, step = pixelsize, length = pixelnumx + 1)),
        collect(range(offsety, step = pixelsize, length = pixelnumy + 1)),
        pixelnumx,
        pixelnumy,
        pixelsize,
        pixelsize^2 * exposure,
        na,
        refractiveindex,
        wavelength,
        psf,
    )
end

struct Priors
    photostate::Categorical
    location::MvNormal
    bleachrate::Gamma
    diffusion::InverseGamma
    # emissionrate::Gamma
    background::Gamma
    gain::InverseGamma
    load::Bernoulli
    Priors(;
        photostate_p::AbstractVector{<:Real} = [1, 0],
        location_μx::Real = 0,
        location_σx::Real = 0,
        location_μy::Real = 0,
        location_σy::Real = 0,
        location_μz::Real = 0,
        location_σz::Real = 0,
        bleachrate_ϕ::Real = 1,
        bleachrate_ψ::Real = 1,
        diffusion_ϕ::Real = 1,
        diffusion_χ::Real = 1,
        background_ϕ::Real = 1,
        background_ψ::Real = 1,
        gain_ϕ::Real = 1,
        gain_χ::Real = 1,
        load_p::Real = 0.1,
    ) = new(
        Categorical(photostate_p),
        MvNormal(
            [location_μx, location_μy, location_μz],
            Diagonal([location_σx, location_σy, location_σz]),
        ),
        Gamma(bleachrate_ϕ, bleachrate_ψ / bleachrate_ϕ),
        InverseGamma(diffusion_ϕ, diffusion_ϕ * diffusion_χ),
        Gamma(background_ϕ, background_ψ / background_ϕ),
        InverseGamma(gain_ϕ, gain_ϕ * gain_χ),
        Bernoulli(load_p),
    )
end

struct Video
    data::Array{Float64,3}
    params::ExperimentalParameters
end

struct GroundTruth
    particle_num::Int
    tracks::Array{Float64,3}
    photostate::Array{Int8,3}
    diffus_coeff::Real
    # emissionrate::Float64
    background::Float64
    gain::Float64
    times::Vector{Float64}
    length_per_exposure::Int
    emitterPSF::Array{Float64,3}
end

mutable struct Sample
    spatial_loc::Array{Float64,3}
    photostate::Array{Int8,3}
    load::Vector{Bool}
    emissionrate::Float64
    background::Float64
end

mutable struct MarkovChain
    x::Float64
end