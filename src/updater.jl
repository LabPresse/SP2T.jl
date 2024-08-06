function simulate!(
    data::Data{T};
    diffusivity::Real,
    brightness::Real,
    nemitters::Integer,
    μ = nothing,
    σ = [0, 0, 0],
) where {T}
    x = Array{T}(undef, size(data.frames, 3), 3, nemitters)
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
    h::T,
    data::Data{T},
    aux::AuxiliaryVariables{T},
) where {T}
    xᵒⁿ, yᵒⁿ = ontracks(x, M)
    propose!(yᵒⁿ, xᵒⁿ, x.perturbsize)
    pxcounts!(aux.U, xᵒⁿ, h, data)
    pxcounts!(aux.V, yᵒⁿ, h, data)
    return x
end

function update_odd!(
    𝐱::AbstractArray{T,3},
    𝐲::AbstractArray{T,3},
    Δ𝐱²::AbstractArray{T,3},
    Δ𝐲²::AbstractArray{T,3},
    D::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΔΔx²::AbstractArray{T,3},
    aux::AuxiliaryVariables{T},
) where {T}
    Δlogℒ = aux.Sᵥ
    oddΔlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², D, ΔΔx², aux.ΣΔΔ𝐱²)
    @views begin
        logr[1:2:end] .+= Δlogℒ[1:2:end]
        accept[1:2:end] .= logr[1:2:end] .> 0
    end
    copyidxto!(𝐱, 𝐲, accept)
end

function update_even!(
    𝐱::AbstractArray{T,3},
    𝐲::AbstractArray{T,3},
    Δ𝐱²::AbstractArray{T,3},
    Δ𝐲²::AbstractArray{T,3},
    D::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΔΔx²::AbstractArray{T,3},
    aux::AuxiliaryVariables{T},
) where {T}
    Δlogℒ = aux.Sᵥ
    evenΔlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², D, ΔΔx², aux.ΣΔΔ𝐱²)
    @views begin
        logr[2:2:end] .+= Δlogℒ[2:2:end]
        accept[2:2:end] .= logr[2:2:end] .> 0
    end
    copyidxto!(𝐱, 𝐲, accept)
end

function update_ontracks!(
    x::BrownianTracks,
    M::Integer,
    D::T,
    h::T,
    data::Data{T},
    𝑇::T,
    aux::AuxiliaryVariables{T},
) where {T}
    MHinit!(x)
    xᵒⁿ, yᵒⁿ = ontracks(x, M)
    Δxᵒⁿ², Δyᵒⁿ², ΔΔxᵒⁿ² = displacements(aux, M)
    propose!(yᵒⁿ, xᵒⁿ, x)
    pxcounts!(aux.U, xᵒⁿ, h, data)
    pxcounts!(aux.V, yᵒⁿ, h, data)
    Δlogℒ!(aux.Sᵥ, data.frames, aux.U, aux.V, data.filter, data.batchsize, aux.Sₐ)
    x.logratio .+= anneal!(aux.Sᵥ, 𝑇)
    addΔlogπ₁!(x.logratio, xᵒⁿ, yᵒⁿ, x.prior)
    update_odd!(xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ², D, x.logratio, x.accepted, ΔΔxᵒⁿ², aux)
    update_even!(xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ², D, x.logratio, x.accepted, ΔΔxᵒⁿ², aux)
    counter!(x)
    return x
end

function update_offtracks!(x::BrownianTracks, M::Integer, D::Real)
    @views xᵒᶠᶠ = x.value[:, :, M+1:end]
    μ, σ = _params(x.prior)
    simulate!(xᵒᶠᶠ, μ, σ, D)
    return x
end

function update!(
    D::Diffusivity,
    x::AbstractArray{T,3},
    𝑇::T,
    Δx²::AbstractArray{T,3},
) where {T}
    diff²!(Δx², x)
    setparams!(D, Δx², 𝑇)
    return sample!(D)
end

function update!(
    M::NEmitters,
    x::AbstractArray{T,3},
    h::T,
    data::Data{T},
    𝑇::T,
    A::AuxiliaryVariables{T},
) where {T}
    setlogℒ!(M, x, h, data, A)
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
    data::Data{T},
    niters::Integer,
    prev_niters::Integer,
    A::AuxiliaryVariables{T},
) where {T}
    @showprogress 1 "Computing..." for iter = 1:niters
        𝑇 = temperature(chain, iter)
        update_offtracks!(x, M.value, D.value)
        if anyactive(M)
            update_ontracks!(x, M.value, D.value, h.value, data, 𝑇, A)
            permuteemitters!(x.value, x.valueᵖ, M.value)
        end
        update!(D, x.value, 𝑇, A.Δ𝐱²)
        update!(M, x.value, h.value, data, 𝑇, A)
        if iter % saveperiod(chain) == 0
            log𝒫, logℒ = log𝒫logℒ(x, M, D, h, data, A)
            push!(chain.samples, Sample(x, M, D, h, iter + prev_niters, 𝑇, log𝒫, logℒ))
            isfull(chain) && shrink!(chain)
        end
    end
    return chain
end

function runMCMC!(
    chain::Chain,
    x::BrownianTracks,
    M::NEmitters,
    D::Diffusivity,
    h::Brightness,
    data::Data,
    niters::Integer,
)
    prev_niters = chain.samples[end].iteration
    A = AuxiliaryVariables(x.value, size(data.frames))
    A.U .= data.darkcounts
    M.logℒ[1] = logℒ(data.frames, A.U, data.filter, data.batchsize, A.Sₐ, A.Sᵥ)
    runMCMC!(chain, x, M, D, h, data, niters, prev_niters, A)
end

function runMCMC(;
    tracks::BrownianTracks,
    nemitters::NEmitters,
    diffusivity::Diffusivity,
    brightness::Brightness,
    data::Data,
    niters::Integer = 1000,
    sizelimit::Integer = 1000,
    annealing::Union{AbstractAnnealing,Nothing} = nothing,
)
    isnothing(annealing) && (annealing = NoAnnealing{typeof(data.period)}())
    chain =
        Chain([Sample(tracks, nemitters, diffusivity, brightness)], sizelimit, annealing)
    runMCMC!(chain, tracks, nemitters, diffusivity, brightness, data, niters)
    return chain
end
