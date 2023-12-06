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

