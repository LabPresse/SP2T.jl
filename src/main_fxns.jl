function simulate_sample(;
    param::ExperimentalParameter{FT},
    framecount::Integer,
    emittercount::Integer,
    diffusion_coefficient::Real,
    brightness::Real,
    init_pos_prior::Union{Missing,MultivariateDistribution} = missing,
    device::Device = CPU(),
) where {FT}
    D = FT(diffusion_coefficient)
    if ismissing(init_pos_prior)
        init_pos_prior = default_init_pos_prior(param)
    end
    x = Array{FT,3}(undef, 3, emittercount, framecount)
    simulate!(x, init_pos_prior, D, param.period, device)
    return Sample(x, D, FT(brightness))
end