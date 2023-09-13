function get_xᵖ(
    xᵒ::AbstractArray{FT,3},
    σ::AbstractVector{FT},
    param::ExperimentalParameter{FT},
) where {FT<:AbstractFloat}
    xᵖ = xᵒ .+ σ .* CUDA.randn(size(xᵒ)...)
    Gᵖ = simulate_G(xᵖ, param.pxboundsx, param.pxboundsy, param.PSF)
    return xᵖ, Gᵖ
end

function get_Δx²(
    xᵖ::AbstractMatrix{FT},
    xᵒ::AbstractMatrix{FT},
    x::AbstractArray{FT},
) where {FT<:AbstractFloat}
    return sum((xᵖ .- x) .^ 2 .- (xᵒ .- x) .^ 2)
end

function diff_lnℒ_x(
    w::AbstractArray{Bool},
    h::AbstractFloat,
    Gᵖ::AbstractArray,
    Gᵒ::AbstractArray,
    F::AbstractMatrix,
)
    uᵖ = F .+ h .* Gᵖ
    uᵒ = F .+ h .* Gᵒ
    lnℒ_diff = w .* (logexpm1.(uᵖ) .- logexpm1.(uᵒ)) .- (uᵖ .- uᵒ)
    return sum(lnℒ_diff, dims = (1, 2))
    #? replace sum with dot?
end

diff_ln𝒫_x₁(
    xᵖ₁::AbstractMatrix{FT},
    xᵒ₁::AbstractMatrix{FT},
    μ::AbstractVector{FT},
    σ²::AbstractVector{FT},
) where {FT<:AbstractFloat} = sum(((xᵒ₁ .- μ) .^ 2 - (xᵖ₁ .- μ) .^ 2) ./ (2 .* σ²))

function diff_ln𝒫_x(
    xᵖ::AbstractArray{FT,3},
    xᵒ::AbstractArray{FT,3},
    n::Integer,
    τ::FT,
    sum_Δxᵒ²::FT,
    𝒫_D::InverseGamma{FT},
    𝒫_x::MvNormal,
) where {FT<:AbstractFloat}
    ~, B, N = size(xᵒ)
    ϕ, ϕχτ4 = shape(𝒫_D), 4 * τ * scale(𝒫_D)
    sum_Δxᵖ² =
        sum_Δxᵒ² + get_Δx²(
            view(xᵖ, :, :, n),
            view(xᵒ, :, :, n),
            view(xᵒ, :, :, max(1, n - 1):2:min(n + 1, N)),
        )
    diff_ln𝒫 = (-1.5 * B * (N - 1) - ϕ) * log((sum_Δxᵖ² + ϕχτ4) / (sum_Δxᵒ² + ϕχτ4))
    n == 1 && (
        diff_ln𝒫 += diff_ln𝒫_x₁(
            view(xᵖ, :, :, 1),
            view(xᵒ, :, :, 1),
            CuArray(𝒫_x.μ),
            CuArray(diag(𝒫_x.Σ)),
        )
    )
    return diff_ln𝒫, sum_Δxᵖ²
end

function diff_ln𝒫_x(
    xᵖ::AbstractArray{FT,3},
    xᵒ::AbstractArray{FT,3},
    n::Integer,
    τ::FT,
    D::FT,
    𝒫_x::MvNormal,
) where {FT<:AbstractFloat}
    ~, B, N = size(xᵒ)
    Δx² = get_Δx²(
        view(xᵖ, :, :, n),
        view(xᵒ, :, :, n),
        view(xᵒ, :, :, max(1, n - 1):2:min(n + 1, N)),
    )
    diff_ln𝒫 = -Δx² / (4 * D * τ)
    n == 1 && (
        diff_ln𝒫 += diff_ln𝒫_x₁(
            view(xᵖ, :, :, 1),
            view(xᵒ, :, :, 1),
            CuArray(𝒫_x.μ),
            CuArray(diag(𝒫_x.Σ)),
        )
    )
    return diff_ln𝒫
end

get_ln𝓇_x(
    w::AbstractArray{Bool,3},
    G::AbstractArray{FT,3},
    hᵖ::FT,
    hᵒ::FT,
    F::AbstractMatrix{FT},
    𝒫::Gamma{FT},
) where {FT<:AbstractFloat} =
    diff_lnℒ_h(w, G, hᵖ, hᵒ, F) + diff_ln𝒫_h(hᵖ, hᵒ, 𝒫) + diff_ln𝒬_h(hᵖ, hᵒ)

# function sample_on_x(
#     w::AbstractArray{Bool},
#     x::MetropolisHastingsVectorRV,
#     D::RandomVariable,
#     h::AbstractFloat,
#     Gᵒ::AbstractArray{<:AbstractFloat},
#     param::ExperimentalParameter,
# )
#     diff_lnℒ = diff_lnℒ_x(w, h, Gᵖ, Gᵒ, param.darkcounts) |> cpu

#     sum_Δxᵒ² = sum(diff(xᵒ, dims = 3) .^ 2)

#     for n in randperm(N)
#         ln𝓇, sum_Δxᵖ² =
#             diff_ln𝒫_x(view(xᵖ, :, :, n), xᵒ, n, param.period, sum_Δxᵒ², D.𝒫, x.𝒫)
#         ln𝓇 += diff_lnℒ[n]

#         if log(rand()) <= ln𝓇
#             xᵒ[:, :, n] = xᵖ[:, :, n]
#             sum_Δxᵒ² = sum_Δxᵖ²
#             x.count[1] += 1
#         end

#         x.count[2] += 1
#     end

#     Gᵖ = simulate_G(xᵖ, param.pxboundsx, param.pxboundsy, param.PSF)
# end

# function update_on_x!(s::ChainStatus, w::AbstractArray{Bool}, param::ExperimentalParameter)
#     x = s.x
#     B = get_B(s)
#     xᵒ, Gᵒ, τ = view(s.x.value, :, 1:B, :), s.G, param.period
#     sum_Δxᵒ² = sum(diff(xᵒ, dims = 3) .^ 2)
#     xᵖ, Gᵖ = get_xᵖ(xᵒ, CuArray(diag(x.𝒬.Σ)), param)
#     diff_lnℒ = diff_lnℒ_x(w, s.h.value, Gᵖ, Gᵒ, param.darkcounts) |> cpu
#     for n in randperm(param.length)
#         ln𝓊 = log(rand())
#         ln𝓇, sum_Δxᵖ² = diff_ln𝒫_x(xᵖ, xᵒ, n, τ, sum_Δxᵒ², s.D.𝒫, x.𝒫)
#         ln𝓇 += diff_lnℒ[n]
#         if ln𝓇 > ln𝓊
#             xᵒ[:, :, n] .= xᵖ[:, :, n]
#             sum_Δxᵒ² = sum_Δxᵖ²
#             x.count[1] += 1
#         end
#         x.count[2] += 1
#     end
#     #? potential improvemnt
#     s.G = simulate_G(xᵒ, param.pxboundsx, param.pxboundsy, param.PSF)
# end

function update_on_x!(s::ChainStatus, w::AbstractArray{Bool}, param::ExperimentalParameter)
    x = s.x
    B = get_B(s)
    xᵒ, Gᵒ, τ = view(s.x.value, :, 1:B, :), s.G, param.period
    xᵖ, Gᵖ = get_xᵖ(xᵒ, CuArray(diag(x.𝒬.Σ)), param)
    diff_lnℒ = diff_lnℒ_x(w, s.h.value, Gᵖ, Gᵒ, param.darkcounts) |> cpu
    for n in randperm(param.length)
        ln𝓊 = log(rand())
        ln𝓇 = diff_ln𝒫_x(xᵖ, xᵒ, n, τ, s.D.value, x.𝒫)
        ln𝓇 += diff_lnℒ[n]
        if ln𝓇 > ln𝓊
            xᵒ[:, :, n] .= xᵖ[:, :, n]
            x.count[1] += 1
        end
        x.count[2] += 1
    end
    #? potential improvemnt
    s.G = simulate_G(xᵒ, param.pxboundsx, param.pxboundsy, param.PSF)
end

# sample_on_x(
#     w::AbstractArray{Bool},
#     x::MetropolisHastingsVectorRV,
#     D::RandomVariable,
#     h::AbstractFloat,
#     Gᵒ::AbstractArray{<:AbstractFloat},
#     param::ExperimentalParameter,
# )

update_off_x!(s::ChainStatus, prior::Distribution, param::ExperimentalParameter) =
    simulate!(view_off_x(s), prior, s.D, param.period)