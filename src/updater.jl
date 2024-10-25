# function simulate!(
#     data::Data;
#     diffusivity::Real,
#     brightness::Real,
#     nemitters::Integer,
#     μ = nothing,
#     σ = [0, 0, 0],
# )
#     T = typeof(data.period)
#     x = Array{T}(undef, size(data.frames, 3), 3, nemitters)
#     msd = 2 * convert(T, diffusivity) * data.period
#     h = convert(T, brightness) * data.period
#     σ = convert(Vector{T}, σ)
#     isnothing(μ) && (μ = framecenter(data))
#     simulate!(x, μ, σ, msd)
#     groundtruth = Sample(x, msd, h, 0, one(T), zero(T), zero(T))
#     simframes!(data.frames, pxcounts(groundtruth.tracks, groundtruth.brightness, data))
#     return data, groundtruth
# end

function update_odd!(
    𝐱::AbstractArray{T,3},
    𝐲::AbstractArray{T,3},
    Δ𝐱²::AbstractArray{T,3},
    Δ𝐲²::AbstractArray{T,3},
    msd::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΔΔx²::AbstractArray{T,3},
    A::AuxiliaryVariables,
) where {T}
    Δlogℒ = A.Sᵥ
    oddΔlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², msd, ΔΔx², A.ΣΔΔ𝐱²)
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
    msd::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΔΔx²::AbstractArray{T,3},
    A::AuxiliaryVariables,
) where {T}
    Δlogℒ = A.Sᵥ
    evenΔlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², msd, ΔΔx², A.ΣΔΔ𝐱²)
    @views begin
        logr[2:2:end] .+= Δlogℒ[2:2:end]
        accept[2:2:end] .= logr[2:2:end] .> 0
    end
    copyidxto!(𝐱, 𝐲, accept)
end

function update_ontracks!(
    x::Tracks{T},
    M::Integer,
    msd::T,
    h::T,
    data::Data{T},
    𝑇::T,
    A::AuxiliaryVariables{T},
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
    update_odd!(xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ², msd, logr, accep, ΔΔxᵒⁿ², A)
    update_even!(xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ², msd, logr, accep, ΔΔxᵒⁿ², A)
    counter!(x)
    return x
end

function update_offtracks!(x::Tracks, M::Integer, msd::Real)
    @views xᵒᶠᶠ = x.value[:, :, M+1:end]
    μ, σ = params(x.prior)
    simulate!(xᵒᶠᶠ, μ, σ, msd)
    return x
end

function update!(
    msd::MeanSquaredDisplacement{T},
    𝐱::AbstractArray{T,3},
    𝑇::T,
    Δ𝐱²::AbstractArray{T,3},
) where {T}
    diff²!(Δ𝐱², 𝐱)
    return sample!(msd, Δ𝐱², 𝑇)
end

# function update!(
#     msd::MSD{T},
#     𝐱::AbstractArray{T,3},
#     𝑇::T,
#     auxvar::AuxiliaryVariables,
# ) where {T}
#     diff²!(auxvar.Δ𝐱², 𝐱)
#     # setparams!(D, A.Δ𝐱², 𝑇)
#     return sample!(msd, auxvar.Δ𝐱², 𝑇)
# end

function update!(
    M::NEmitters{T},
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

function extend!(
    chain::Chain{T},
    x::Tracks{T},
    M::NEmitters{T},
    msd::MeanSquaredDisplacement{T},
    h::Brightness{T},
    data::Data{T},
    iter::Integer,
    𝑇::T,
    A::AuxiliaryVariables{T},
) where {T}
    if iter % chain.stride == 0
        log𝒫, logℒ = log𝒫logℒ(x, M, msd, h, data, A)
        push!(
            chain.samples,
            Sample(x.value, M.value, msd.value, h.value, iter, 𝑇, log𝒫, logℒ),
        )
        isfull(chain) && shrink!(chain)
    end
    return chain
end

function parametricMCMC!(
    # chain::Chain{T},
    x::Tracks{T},
    M::NEmitters{T},
    msd::MeanSquaredDisplacement{T},
    h::Brightness{T},
    data::Data{T},
    𝑇::T,
    A::AuxiliaryVariables{T},
) where {T}
    # @showprogress 1 "Computing..." for iter = 1:niters
    # 𝑇 = temperature(chain, iter)
    update_ontracks!(x, M.value, msd.value, h.value, data, 𝑇, A)
    update!(msd, view(x.value, :, :, 1:M.value), 𝑇, view(A.Δ𝐱², :, :, 1:M.value))
    # extend!(chain, x, M, msd, h, data, iter, 𝑇, A)
    # end
    return x, msd
end

function nonparametricMCMC!(
    # chain::Chain{T},
    x::Tracks{T},
    M::NEmitters{T},
    msd::MeanSquaredDisplacement{T},
    h::Brightness{T},
    data::Data{T},
    𝑇::T,
    A::AuxiliaryVariables{T},
) where {T}
    # @showprogress 1 "Computing..." for iter = 1:niters
    # 𝑇 = temperature(chain, iter)
    update_offtracks!(x, M.value, msd.value)
    if any(M)
        update_ontracks!(x, M.value, msd.value, h.value, data, 𝑇, A)
        permuteemitters!(x.value, x.valueᵖ, M.value)
    end
    update!(msd, x.value, 𝑇, A.Δ𝐱²)
    update!(M, x.value, h.value, data, 𝑇, A)
    # extend!(chain, x, M, msd, h, data, iter, 𝑇, A)
    # end
    return x, msd, M
end

function runMCMC!(
    chain::Chain{T},
    x::Tracks{T},
    M::NEmitters{T},
    msd::MeanSquaredDisplacement{T},
    h::Brightness{T},
    data::Data{T},
    niters::Integer,
    parametric::Bool,
) where {T}
    prev_niters = chain.samples[end].iteration
    A = AuxiliaryVariables(x.value, size(data.frames))
    A.U .= data.darkcounts
    M.logℒ[1] = logℒ(data, A)
    # if parametric
    #     parametricMCMC!(chain, x, M, msd, h, data, niters, prev_niters, A)
    # else
    #     nonparametricMCMC!(chain, x, M, msd, h, data, niters, prev_niters, A)
    # end

    nextsample! = parametric ? parametricMCMC! : nonparametricMCMC!
    @showprogress 1 "Computing..." for iter in prev_niters .+ (1:niters)
        𝑇 = temperature(chain, iter)
        nextsample!(x, M, msd, h, data, 𝑇, A)
        extend!(chain, x, M, msd, h, data, iter, 𝑇, A)
    end
end

function runMCMC(;
    tracks::Tracks{T},
    nemitters::NEmitters{T},
    diffusivity::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    data::Data{T},
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
