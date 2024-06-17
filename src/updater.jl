function simulate!(
    data::Data;
    diffusivity::Real,
    brightness::Real,
    nemitters::Integer,
    μ = nothing,
    σ = [0, 0, 0],
)
    T = typeof(data.period)
    x = Array{T}(undef, 3, nemitters, size(data.frames, 3))
    D = convert(T, diffusivity) * data.period
    h = convert(T, brightness) * data.period
    σ = convert(Vector{T}, σ)
    isnothing(μ) && (μ = framecenter(data))
    simulate!(x, μ, σ, D)
    groundtruth = Sample(x, D, h, 0, one(T), zero(T), zero(T))
    simframes!(data.frames, pxcounts(groundtruth.tracks, groundtruth.brightness, data))
    return data, groundtruth
end

function propose!(
    x::BrownianTracks,
    M::Integer,
    h::Real,
    data::Data,
    aux::AuxiliaryVariables,
)
    xᵒⁿ, yᵒⁿ = ontracks(x, M)
    propose!(yᵒⁿ, xᵒⁿ, x.perturbsize)
    pxcounts!(aux.U, xᵒⁿ, h, data)
    pxcounts!(aux.V, yᵒⁿ, h, data)
    return x
end

function update_odd!(
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    D::T,
    logr::AbstractArray{T,3},
    accept::AbstractArray{Bool,3},
    ΔΔx²::AbstractArray{T,3},
    aux::AuxiliaryVariables,
) where {T}
    oddΔlogπ!(aux.ΔlogP, x, y, Δx², Δy², D, ΔΔx², aux.ΣΔΔx²)
    @views begin
        logr[1:2:end] .+= aux.ΔlogP[1:2:end]
        # accept_odd!(x, y, accept, logr)
        accept[1:2:end] .= logr[1:2:end] .> 0
    end
    copyidxto!(x, y, accept)
end

function update_even!(
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    D::T,
    logr::AbstractArray{T,3},
    accept::AbstractArray{Bool,3},
    ΔΔx²::AbstractArray{T,3},
    aux::AuxiliaryVariables,
) where {T}
    evenΔlogπ!(aux.ΔlogP, x, y, Δx², Δy², D, ΔΔx², aux.ΣΔΔx²)
    @views begin
        logr[2:2:end] .+= aux.ΔlogP[2:2:end]
        # accept_even!(x, y, accept, logr)
        accept[2:2:end] .= logr[2:2:end] .> 0
    end
    copyidxto!(x, y, accept)
end

function update_ontracks!(
    x::BrownianTracks,
    M::Integer,
    D::T,
    h::T,
    data::Data,
    𝑇::Union{T,Int},
    aux::AuxiliaryVariables,
) where {T}
    MHinit!(x)
    xᵒⁿ, yᵒⁿ = ontracks(x, M)
    Δxᵒⁿ², Δyᵒⁿ², ΔΔxᵒⁿ² = displacements(aux, M)
    propose!(yᵒⁿ, xᵒⁿ, x)
    pxcounts!(aux.U, xᵒⁿ, h, data)
    pxcounts!(aux.V, yᵒⁿ, h, data)
    x.logratio .+= Δlogℒ!(aux.ΔlogP, data.frames, aux.U, aux.V, aux.ΔU, 𝑇)
    addΔlogπ₁!(x.logratio, xᵒⁿ, yᵒⁿ, x.prior)
    update_odd!(xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ², D, x.logratio, x.accepted, ΔΔxᵒⁿ², aux)
    update_even!(xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ², D, x.logratio, x.accepted, ΔΔxᵒⁿ², aux)
    counter!(x)
    # copyidxto!(aux.U, aux.V, x.accepted)
    return x
end

function update_offtracks!(x::BrownianTracks, M::Integer, D::Real)
    @views xᵒᶠᶠ = x.value[:, M+1:end, :]
    μ, σ = _params(x.prior)
    simulate!(xᵒᶠᶠ, μ, σ, D)
    return x
end

function update!(
    D::Diffusivity,
    x::AbstractArray{T,3},
    𝑇::Union{T,Int},
    aux::AuxiliaryVariables,
) where {T}
    diff²!(aux.Δx², x)
    setparams!(D, aux.Δx², 𝑇, aux.𝟙Δx)
    return sample!(D)
end

function update!(
    M::NEmitters,
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    h::T,
    data::Data,
    𝑇::Union{T,Int},
    aux::AuxiliaryVariables,
) where {T}
    shuffletracks!(x, y, M)
    setlogℒ!(M, aux.U, x, h, data, aux.ΔU, aux.𝟙U)
    setlog𝒫!(M, 𝑇)
    sample!(M)
    return M
end

Sample(x::BrownianTracks, M::NEmitters, D::Diffusivity, h::Brightness, i, 𝑇, log𝒫, logℒ) =
    Sample(x.value, M.value, D.value, h.value, i, 𝑇, log𝒫, logℒ)

Sample(x::BrownianTracks, M::NEmitters, D::Diffusivity, h::Brightness) =
    Sample(x.value, M.value, D.value, h.value)

function runMCMC!(
    chain::Chain,
    x::BrownianTracks,
    M::NEmitters,
    D::Diffusivity,
    h::Brightness,
    data::Data,
    niters::Integer,
    prev_niters::Integer,
    aux::AuxiliaryVariables,
)
    @showprogress 1 "Computing..." for iter = 1:niters
        𝑇 = temperature(chain, iter)
        update_offtracks!(x, M.value, D.value)
        anyactive(M) && update_ontracks!(x, M.value, D.value, h.value, data, 𝑇, aux)
        update!(D, x.value, 𝑇, aux)
        update!(M, x.value, x.valueᵖ, h.value, data, 𝑇, aux)
        if iter % saveperiod(chain) == 0
            log𝒫, logℒ = log𝒫logℒ(x, M, D, h, data, aux)
            push!(chain.samples, Sample(x, M, D, h, iter + prev_niters, 𝑇, log𝒫, logℒ))
            isfull(chain) && shrink!(chain)
        end
    end
    return chain
end

AuxiliaryVariables(x::BrownianTracks, data::Data) =
    AuxiliaryVariables(x.value, data.pxboundsx, data.pxboundsy, data.darkcounts)

function runMCMC!(
    chain::Chain,
    x::BrownianTracks,
    M::NEmitters,
    D::Diffusivity,
    h::Brightness,
    data::Data,
    niters,
)
    prev_niters = chain.samples[end].iteration
    aux = AuxiliaryVariables(x, data)
    M.logℒ[1] = _logℒ(data.frames, aux.V, aux.ΔU, aux.𝟙U)
    runMCMC!(chain, x, M, D, h, data, niters, prev_niters, aux)
end

function runMCMC(;
    tracks::BrownianTracks,
    nemitters::NEmitters,
    diffusivity::Diffusivity,
    brightness::Brightness,
    data::Data,
    niters::Integer = 1000,
    sizelimit::Integer = 1000,
    annealing::AbstractAnnealing = NoAnnealing(),
)
    chain =
        Chain([Sample(tracks, nemitters, diffusivity, brightness)], sizelimit, annealing)
    runMCMC!(chain, tracks, nemitters, diffusivity, brightness, data, niters)
    return chain
end
