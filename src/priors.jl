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