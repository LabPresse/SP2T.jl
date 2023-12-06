function simulate_sample(;
    param::ExperimentalParameter{FT},
    emitter_number::Integer,
    diffusion_coefficient::Real,
    emission_rate::Real,
    init_pos_prior::Union{Missing,MultivariateDistribution} = missing,
    device::Device = CPU(),
) where {FT}
    B, N, τ, D, h = emitter_number,
    param.length,
    param.period,
    FT(diffusion_coefficient),
    FT(emission_rate)
    if ismissing(init_pos_prior)
        init_pos_prior = default_init_pos_prior(param)
    end
    x = Array{FT,3}(undef, 3, B, N)
    simulate!(x, init_pos_prior, D, τ, device)
    return Sample(x, D, h)
end