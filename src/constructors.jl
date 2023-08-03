Prior(params::ExperimentalParameter) = Prior(
    μₓ = [params.pxnumx * params.pxsize / 2, params.pxnumy * params.pxsize / 2, 0],
    σₓ = [params.pxsize * 2, params.pxsize * 2, 0],
)

Sample(s::FullSample) = Sample(s.x[:, s.b, :], s.D, s.h, s.F, s.i, s.T, s.logℙ)

# Sample(s::FullSample) = Sample(s.x[:, 1:get_B(s), :], s.D, s.h, s.F, s.i, s.T, s.logℙ)

function FullSample(s::Sample{FT}, M::Integer) where {FT<:AbstractFloat}
    (~, B, N) = size(s.x)
    M < B && error("Weak limit is too small!")
    b = BitVector(zeros(Bool, 5))
    b[1:B] .= true
    x = Array{FT,3}(undef, 3, M, N)
    x[:, 1:B, :] = s.x
    return FullSample(b, x, s.D, s.h, s.F, iszero(s.i) ? 1 : s.i, s.T, s.logℙ)
    #TODO initialize T and logℙ better
end

function Sample(
    FloatType::DataType;
    param::ExperimentalParameter,
    prior::Prior,
    emitter_number::Integer,
    diffusion_coefficient::Real,
    emission_rate::Real,
    background_flux::Matrix{<:Real},
)
    ftypeof(param) ≡ FloatType ||
        @warn "The float type in the argument is different from that of the experimental parameter!"
    (B, N, T, D, h, F) = (
        emitter_number,
        param.length,
        FloatType(param.period),
        FloatType(diffusion_coefficient),
        FloatType(emission_rate),
        FloatType.(background_flux),
    )
    x = Array{FloatType,3}(undef, 3, B, N)
    simulate!(x, prior.x, D, T)
    return Sample(x, D, h, F)
end

function Video(p::ExperimentalParameter, s::Sample)
    ftypeof(p) ≡ ftypeof(s) ||
        @warn "Float type mismatch between the experimental parameter and the sample!"
    g = Array{ftypeof(p),3}(undef, p.pxnumx, p.pxnumy, p.length)
    d = BitArray{3}(undef, p.pxnumx, p.pxnumy, p.length)
    simulate!(g, s.x, p.pxboundsx, p.pxboundsy, p.PSF)
    simulate!(d, g, s.h, s.F, p.exposure, p.pxareatimesexposure)
    return Video(d, p)
end

function Chain(;
    initial_guess::Sample{FT},
    max_emitter_num::Integer,
    prior::Prior,
    sizelimit::Integer,
    annealing::Union{Annealing,Nothing} = nothing,
) where {FT<:AbstractFloat}
    if isnothing(annealing)
        annealing = PolynomialAnnealing{FT}()
    elseif !(ftypeof(initial_guess) ≡ ftypof(annealing))
        @warn "Float type missmatch between initial_guss and annealing!"
    end
    return Chain{FT}(
        FullSample(initial_guess, max_emitter_num),
        [initial_guess],
        annealing,
        Acceptances(),
        prior,
        1,
        sizelimit,
    )
end