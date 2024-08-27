function simulate!(
    data::Data;
    diffusivity::Real,
    brightness::Real,
    nemitters::Integer,
    μ = nothing,
    σ = [0, 0, 0],
)
    T = typeof(data.period)
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

# function propose!(
#     x::BrownianTracks,
#     M::Integer,
#     h::T,
#     data::Data,
#     aux::AuxiliaryVariables,
# ) where {T}
#     xᵒⁿ, yᵒⁿ = ontracks(x, M)
#     propose!(yᵒⁿ, xᵒⁿ, x.perturbsize)
#     pxcounts!(aux.U, xᵒⁿ, h, data)
#     pxcounts!(aux.V, yᵒⁿ, h, data)
#     return x
# end

function update_odd!(
    𝐱::AbstractArray{T,3},
    𝐲::AbstractArray{T,3},
    Δ𝐱²::AbstractArray{T,3},
    Δ𝐲²::AbstractArray{T,3},
    D::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΔΔx²::AbstractArray{T,3},
    A::AuxiliaryVariables,
) where {T}
    Δlogℒ = A.Sᵥ
    oddΔlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², D, ΔΔx², A.ΣΔΔ𝐱²)
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
    A::AuxiliaryVariables,
) where {T}
    Δlogℒ = A.Sᵥ
    evenΔlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², D, ΔΔx², A.ΣΔΔ𝐱²)
    @views begin
        logr[2:2:end] .+= Δlogℒ[2:2:end]
        accept[2:2:end] .= logr[2:2:end] .> 0
    end
    copyidxto!(𝐱, 𝐲, accept)
end

function update_ontracks!(
    x::Tracks,
    M::Integer,
    D::T,
    h::T,
    data::Data,
    𝑇::T,
    A::AuxiliaryVariables,
) where {T}
    MHinit!(x)
    xᵒⁿ, yᵒⁿ = ontracks(x, M)
    Δxᵒⁿ², Δyᵒⁿ², ΔΔxᵒⁿ² = displacements(A, M)
    logr, accep = x.logratio, x.accepted
    propose!(yᵒⁿ, xᵒⁿ, x.perturbsize)
    pxcounts!(A.U, xᵒⁿ, h, data)
    pxcounts!(A.V, yᵒⁿ, h, data)
    Δlogℒ!(data, A)
    logr .+= anneal!(A.Sᵥ, 𝑇)
    addΔlogπ₁!(logr, xᵒⁿ, yᵒⁿ, x.prior)
    update_odd!(xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ², D, logr, accep, ΔΔxᵒⁿ², A)
    update_even!(xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ², D, logr, accep, ΔΔxᵒⁿ², A)
    counter!(x)
    return x
end

function update_offtracks!(x::Tracks, M::Integer, D::Real)
    @views xᵒᶠᶠ = x.value[:, :, M+1:end]
    μ, σ = _params(x.prior)
    simulate!(xᵒᶠᶠ, μ, σ, D)
    return x
end

function update!(
    D::Diffusivity{T},
    x::AbstractArray{T,3},
    𝑇::T,
    A::AuxiliaryVariables,
) where {T}
    diff²!(A.Δ𝐱², x)
    setparams!(D, A.Δ𝐱², 𝑇)
    return sample!(D)
end

function update!(
    M::NEmitters,
    x::AbstractArray{T,3},
    h::T,
    data::Data,
    𝑇::T,
    A::AuxiliaryVariables,
) where {T}
    setlogℒ!(M, x, h, data, A)
    setlog𝒫!(M, 𝑇)
    sample!(M)
    return M
end

function parametricMCMC!(
    chain::Chain,
    x::Tracks,
    M::NEmitters,
    D::Diffusivity{T},
    h::Brightness{T},
    data::Data,
    niters::Integer,
    prev_niters::Integer,
    A::AuxiliaryVariables,
) where {T}
    @showprogress 1 "Computing..." for iter = 1:niters
        𝑇 = temperature(chain, iter)
        update_ontracks!(x, M.value, D.value, h.value, data, 𝑇, A)
        update!(D, x.value, 𝑇, A)
        if iter % saveperiod(chain) == 0
            log𝒫, logℒ = log𝒫logℒ(x, M, D, h, data, A)
            push!(
                chain.samples,
                Sample(
                    x.value,
                    M.value,
                    D.value,
                    h.value,
                    iter + prev_niters,
                    𝑇,
                    log𝒫,
                    logℒ,
                ),
            )
            isfull(chain) && shrink!(chain)
        end
    end
    return chain
end

function nonparametricMCMC!(
    chain::Chain,
    x::Tracks,
    M::NEmitters,
    D::Diffusivity{T},
    h::Brightness{T},
    data::Data,
    niters::Integer,
    prev_niters::Integer,
    A::AuxiliaryVariables,
) where {T}
    @showprogress 1 "Computing..." for iter = 1:niters
        𝑇 = temperature(chain, iter)
        update_offtracks!(x, M.value, D.value)
        if anyactive(M)
            update_ontracks!(x, M.value, D.value, h.value, data, 𝑇, A)
            permuteemitters!(x.value, x.valueᵖ, M.value)
        end
        update!(D, x.value, 𝑇, A)
        update!(M, x.value, h.value, data, 𝑇, A)
        if iter % saveperiod(chain) == 0
            log𝒫, logℒ = log𝒫logℒ(x, M, D, h, data, A)
            push!(
                chain.samples,
                Sample(
                    x.value,
                    M.value,
                    D.value,
                    h.value,
                    iter + prev_niters,
                    𝑇,
                    log𝒫,
                    logℒ,
                ),
            )
            isfull(chain) && shrink!(chain)
        end
    end
    return chain
end

function runMCMC!(
    chain::Chain,
    x::Tracks,
    M::NEmitters,
    D::Diffusivity{T},
    h::Brightness{T},
    data::Data,
    niters::Integer,
    parametric::Bool,
) where {T}
    prev_niters = chain.samples[end].iteration
    A = AuxiliaryVariables(x.value, size(data.frames))
    A.U .= data.darkcounts
    M.logℒ[1] = logℒ(data, A)
    if parametric
        parametricMCMC!(chain, x, M, D, h, data, niters, prev_niters, A)
    else
        nonparametricMCMC!(chain, x, M, D, h, data, niters, prev_niters, A)
    end
end

function runMCMC(;
    tracks::Tracks,
    nemitters::NEmitters,
    diffusivity::Diffusivity{T},
    brightness::Brightness{T},
    data::Data,
    niters::Integer = 1000,
    sizelimit::Integer = 1000,
    annealing::Union{AbstractAnnealing{T},Nothing} = nothing,
    parametric::Bool = false,
) where {T}
    isnothing(annealing) && (annealing = ConstantAnnealing{T}(1))
    chain = Chain(
        [Sample(tracks.value, nemitters.value, diffusivity.value, brightness.value)],
        sizelimit,
        annealing,
    )
    runMCMC!(chain, tracks, nemitters, diffusivity, brightness, data, niters, parametric)
    return chain
end
