# function update!(
#     diffusivity::Diffusivity,
#     tracks::BrownianTracks,
#     chainparams::ChainParameters,
# )
#     setΔx²!(tracks)
#     setparams!(diffusivity, tracks.Δx², chainparams.temperature)
#     return sample!(diffusivity)
# end

function propose!(
    x::BrownianTracks,
    M::Integer,
    h::Real,
    params::ExperimentalParameters,
    aux::AuxiliaryVariables,
)
    @views xᵒⁿ, yᵒⁿ = x.value[:, 1:M, :], x.valueᵖ[:, 1:M, :]
    propose!(yᵒⁿ, xᵒⁿ, x.perturbsize)
    pxcounts!(aux.U, xᵒⁿ, h, params)
    pxcounts!(aux.Uᵖ, yᵒⁿ, h, params)
    return x
end

frame_Δlogℒ!(
    x::BrownianTracks,
    W::AbstractArray{<:Integer,3},
    𝑇::Real,
    aux::AuxiliaryVariables,
) = frame_Δlogℒ!(x.logratio, W, aux.U, aux.Uᵖ, aux.ΔU, 𝑇)

function accept!(x::BrownianTracks, D::Real, M::Integer, aux::AuxiliaryVariables)
    @views xᵒⁿ, yᵒⁿ, Δxᵒⁿ², Δyᵒⁿ² =
        x.value[:, 1:M, :], x.valueᵖ[:, 1:M, :], aux.Δx²[:, 1:M, :], aux.Δxᵖ²[:, 1:M, :]
    # acc = tracks.accepted
    # update odd frame indices
    diff²!(Δxᵒⁿ², xᵒⁿ)
    diff²!(Δyᵒⁿ², xᵒⁿ, yᵒⁿ)
    add_odd_ΔΔx²!(x.logratio, Δxᵒⁿ², Δyᵒⁿ², D)

    @views x.accepted[:, :, 1:2:end] .=
        x.logratio[:, :, 1:2:end] .> x.logrands[:, :, 1:2:end]
    # oddaccept!(tracks)
    copyidxto!(xᵒⁿ, yᵒⁿ, x.accepted)
    # update even frame indices
    diff²!(Δxᵒⁿ², xᵒⁿ)
    diff²!(Δyᵒⁿ², yᵒⁿ, xᵒⁿ)
    add_even_ΔΔx²!(x.logratio, Δxᵒⁿ², Δyᵒⁿ², D)
    @views x.accepted[:, :, 2:2:end] .=
        x.logratio[:, :, 2:2:end] .> x.logrands[:, :, 2:2:end]
    # evenaccept!(tracks)
    copyidxto!(xᵒⁿ, yᵒⁿ, x.accepted)
    return x
end

function update_ontracks!(
    x::BrownianTracks,
    M::Integer,
    D::T,
    h::T,
    W::AbstractArray{<:Integer,3},
    params::ExperimentalParameters,
    𝑇::Union{T,Int},
    aux::AuxiliaryVariables,
) where {T}
    MHinit!(x)
    propose!(x, M, h, params, aux)
    frame_Δlogℒ!(x, W, 𝑇, aux)
    add_Δlog𝒫!(x, M)
    accept!(x, D, M, aux)
    update_counter!(x)
    copyidxto!(aux.U, aux.Uᵖ, x.accepted)
    return x
end

# function update_offtracks!(tracks::BrownianTracks, M, D)
#     @views x, Δx² = tracks.x[:, M+1:end, :], tracks.Δx²[:, M+1:end, :]
#     μ, σ = _params(tracks.prior)
#     simulate!(x, Δx², μ, σ, D)
#     return tracks
# end

function update_offtracks!(x::BrownianTracks, M, D)
    @views xᵒᶠᶠ = x.value[:, M+1:end, :]
    μ, σ = _params(x.prior)
    simulate!(xᵒᶠᶠ, μ, σ, D)
    return x
end

# function update!(
#     x::BrownianTracks,
#     M,
#     D,
#     h,
#     W::AbstractArray{<:Integer,3},
#     params::ExperimentalParameters,
#     𝑇,
#     aux::AuxiliaryVariables,
# )
#     update_offtracks!(x, M, D)
#     update_ontracks!(x, M, D, h, W, params, 𝑇, aux)
#     return x
# end

function update!(
    D::Diffusivity,
    x::AbstractArray{T,3},
    𝑇::Union{T,Int},
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
    W::AbstractArray{<:Integer,3},
    params::ExperimentalParameters,
    𝑇::Union{T,Int},
    aux::AuxiliaryVariables,
    y::AbstractArray{T,3},
) where {T}
    shuffletracks!(x, M, y)
    setlogℒ!(
        M,
        aux.Uᵖ,
        aux.U,
        W,
        x,
        h,
        params.darkcounts,
        params.pxboundsx,
        params.pxboundsy,
        params.PSF,
        aux.ΔU,
    )
    setlog𝒫!(M, 𝑇)
    sample!(M)
    return M
end

# Sample(
#     tracks::BrownianTracks,
#     nemitters::NEmitters,
#     diffusivity::Diffusivity,
#     brightness::Brightness,
#     chainparams::ChainParameters,
# ) = Sample(
#     Array(tracks.x[:, 1:nemitters.M, :]),
#     diffusivity.D,
#     brightness.h,
#     chainparams.iteration,
#     chainparams.temperature,
#     chainparams.log𝒫,
#     chainparams.logℒ,
# )

Sample(x::BrownianTracks, M::NEmitters, D::Diffusivity, h::Brightness, i, 𝑇, log𝒫, logℒ) =
    Sample(x.value, M.value, D.value, h.value, i, 𝑇, log𝒫, logℒ)

Sample(x::BrownianTracks, M::NEmitters, D::Diffusivity, h::Brightness) =
    Sample(x.value, M.value, D.value, h.value)

# isfull(v::AbstractVector{<:Sample}, sizelimit) = length(v) == sizelimit

# shrink!(v::AbstractVector{<:Sample}) = deleteat!(v, 2:2:lastindex(v))

# function extend!(
#     v::AbstractVector{<:Sample},
#     bt::BrownianTracks,
#     n::NEmitters,
#     d::Diffusivity,
#     b::Brightness,
#     i::Integer,
#     temperature,
#     log𝒫,
#     logℒ,
# )
#     push!(v, Sample(bt.x, n.M, d.D, b.h, i, temperature, log𝒫, logℒ))
#     isfull(v, sizelimit) && shrink!(v)
#     return v
# end

# function update!(
#     x::BrownianTracks,
#     M::NEmitters,
#     D::Diffusivity,
#     h::Brightness,
#     W::AbstractArray{<:Integer,3},
#     params::ExperimentalParameters,
#     𝑇,
#     aux::AuxiliaryVariables,
# )
#     # update!(x, M.value, D.value, h.value, W, params, 𝑇, aux)
#     update_offtracks!(x, M.value, D.value)
#     update_ontracks!(x, M.value, D.value, h.value, W, params, 𝑇, aux)
#     update!(D, x.value, 𝑇, aux.Δx²)
#     update!(M, x.value, h.value, W, params, 𝑇, aux)
#     return nothing
# end

function runMCMC!(
    chain::Chain,
    x::BrownianTracks,
    M::NEmitters,
    D::Diffusivity,
    h::Brightness,
    W::AbstractArray{<:Integer,3},
    params::ExperimentalParameters,
    # chainparams::ChainParameters,
    niters::Integer,
    prev_niters::Integer,
    aux::AuxiliaryVariables,
)
    @showprogress 1 "Computing..." for iter = 1:niters
        𝑇 = temperature(chain, iter)
        update_offtracks!(x, M.value, D.value)
        update_ontracks!(x, M.value, D.value, h.value, W, params, 𝑇, aux)
        update!(D, x.value, 𝑇, aux.Δx²)
        # update!(M, x.value, h.value, W, params, 𝑇, aux, x.valueᵖ)
        if iter % saveperiod(chain) == 0
            log𝒫, logℒ = log𝒫logℒ(x, M, D, h, W, params, aux)
            push!(chain.samples, Sample(x, M, D, h, iter + prev_niters, 𝑇, log𝒫, logℒ))
            isfull(chain) && shrink!(chain)
        end
    end
    return chain
end

AuxiliaryVariables(tracks::BrownianTracks, params::ExperimentalParameters) =
    AuxiliaryVariables(tracks.value, params.pxboundsx, params.pxboundsy)

function runMCMC!(
    chain::Chain,
    x::BrownianTracks,
    M::NEmitters,
    D::Diffusivity,
    h::Brightness,
    W::AbstractArray{<:Integer,3},
    params::ExperimentalParameters,
    # chainparams::ChainParameters,
    niters,
)
    prev_niters = chain.samples[end].iteration
    aux = AuxiliaryVariables(x.value, params.pxboundsx, params.pxboundsy)
    runMCMC!(chain, x, M, D, h, W, params, niters, prev_niters, aux)
end

function runMCMC(;
    tracks::BrownianTracks,
    nemitters::NEmitters,
    diffusivity::Diffusivity,
    brightness::Brightness,
    frames::AbstractArray{<:Integer,3},
    params::ExperimentalParameters,
    # chainparams::ChainParameters,
    niters::Integer = 1000,
    sizelimit::Integer = 1000,
    annealing::AbstractAnnealing = NoAnnealing(),
)
    chain =
        Chain([Sample(tracks, nemitters, diffusivity, brightness)], sizelimit, annealing)
    runMCMC!(chain, tracks, nemitters, diffusivity, brightness, frames, params, niters)
    return chain
end
