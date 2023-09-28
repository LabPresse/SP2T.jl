function get_xᵖ(
    xᵒ::AbstractArray{FT,3},
    𝒬::MvNormal,
    param::ExperimentalParameter{FT},
    ::GPU,
) where {FT<:AbstractFloat}
    xᵖ = xᵒ .+ sqrt.(CuArray(diag(𝒬.Σ))) .* CUDA.randn(size(xᵒ)...)
    Gᵖ = get_pxPSF(xᵖ, param.pxboundsx, param.pxboundsy, param.PSF)
    return xᵖ, Gᵖ
end

function get_xᵖ(
    xᵒ::AbstractArray{FT,3},
    𝒬::MvNormal,
    param::ExperimentalParameter{FT},
    ::CPU,
) where {FT<:AbstractFloat}
    xᵖ = xᵒ .+ sqrt.(viewdiag(𝒬.Σ)) .* randn(size(xᵒ)...)
    Gᵖ = get_pxPSF(xᵖ, param.pxboundsx, param.pxboundsy, param.PSF)
    return xᵖ, Gᵖ
end

get_ΔΔx²(
    xᵖ::AbstractMatrix{FT},
    xᵒ::AbstractMatrix{FT},
    neighbourx::AbstractArray{FT},
) where {FT<:AbstractFloat} = sum((xᵒ .- neighbourx) .^ 2 .- (xᵖ .- neighbourx) .^ 2)

function get_Δlnℒ_x(
    w::AbstractArray{Bool,3},
    Gᵖ::AbstractArray{FT,3},
    Gᵒ::AbstractArray{FT,3},
    hτ::FT,
    F::AbstractMatrix{FT},
) where {FT<:AbstractFloat}
    uᵖ = F .+ hτ .* Gᵖ
    uᵒ = F .+ hτ .* Gᵒ
    Δlnℒ = uᵒ .- uᵖ
    Δlnℒ[w] .+= logexpm1.(uᵖ[w]) .- logexpm1.(uᵒ[w])
    return sum(Δlnℒ, dims = (1, 2))
end

neighbour_indices(n::Integer, N::Integer) = n == 1 ? 2 : n-1:2:min(n + 1, N)

view_neighbour(x::AbstractArray{<:AbstractFloat,3}, n::Integer, N::Integer) =
    view(x, :, :, neighbour_indices(n, N))

get_Δln𝒫_x₁(
    xᵖ::AbstractMatrix{FT},
    xᵒ::AbstractMatrix{FT},
    𝒫::MvNormal,
    ::GPU,
) where {FT<:AbstractFloat} = sum(
    ((xᵒ .- CuArray(𝒫.μ)) .^ 2 - (xᵖ .- CuArray(𝒫.μ)) .^ 2) ./ (2 .* CuArray(diag(𝒫.Σ))),
)

get_Δln𝒫_x₁(
    xᵖ::AbstractMatrix{FT},
    xᵒ::AbstractMatrix{FT},
    𝒫::MvNormal,
    ::CPU,
) where {FT<:AbstractFloat} =
    sum(((xᵒ .- 𝒫.μ) .^ 2 - (xᵖ .- 𝒫.μ) .^ 2) ./ (2 .* viewdiag(𝒫.Σ)))

function get_Δln𝒫_x(
    xᵖ::AbstractVecOrMat{FT},
    xᵒ::AbstractVecOrMat{FT},
    xᶜ::AbstractArray{FT},
    fourDτ::FT,
    isfirstframe::Bool,
    𝒫::MvNormal,
    device::Device,
) where {FT<:AbstractFloat}
    Δln𝒫 = get_ΔΔx²(xᵖ, xᵒ, xᶜ) / fourDτ
    isfirstframe && (Δln𝒫 += get_Δln𝒫_x₁(xᵖ, xᵒ, 𝒫, device))
    return Δln𝒫
end

function update_on_x!(
    s::ChainStatus,
    w::AbstractArray{Bool},
    param::ExperimentalParameter,
    device::Device,
)
    N, F = param.length, param.darkcounts
    hτ, fourDτ = (s.h.value, 4 * s.D.value) .* param.period
    𝒫, 𝒬, counter = s.x.𝒫, s.x.𝒬, view(s.x.counter, :, 2)
    xᵒ, Gᵒ = view_on_x(s), s.G
    xᵖ, Gᵖ = get_xᵖ(xᵒ, 𝒬, param, device)
    Δlnℒ = get_Δlnℒ_x(w, Gᵖ, Gᵒ, hτ, F)
    Δlnℒ isa CuArray && (Δlnℒ = Array(Δlnℒ))
    accepted = BitVector(undef, N)
    @inbounds for n in randperm(N)
        xᵖₙ, xᵒₙ, xⁿₙ = view(xᵖ, :, :, n), view(xᵒ, :, :, n), view_neighbour(xᵒ, n, N)
        ln𝓇 = Δlnℒ[n] + get_Δln𝒫_x(xᵖₙ, xᵒₙ, xⁿₙ, fourDτ, n == 1, 𝒫, device)
        accepted[n] = ln𝓇 > log(rand())
        accepted[n] && (xᵒₙ .= xᵒₙ)
    end
    counter .+= count(accepted), N
    Gᵒ[:, :, accepted] .= view(Gᵖ, :, :, accepted)
end

update_off_x!(s::ChainStatus, param::ExperimentalParameter, device::Device) =
    simulate!(view_off_x(s), s.x.𝒫, s.D.value, param.period, device)

function update_x!(
    s::ChainStatus,
    w::AbstractArray{Bool},
    param::ExperimentalParameter,
    device::Device,
)
    update_off_x!(s::ChainStatus, param::ExperimentalParameter, device::Device)
    update_on_x!(
        s::ChainStatus,
        w::AbstractArray{Bool},
        param::ExperimentalParameter,
        device::Device,
    )
    return s
end

# function get_Δln𝒫_x(
#     xᵖ::AbstractArray{FT,3},
#     xᵒ::AbstractArray{FT,3},
#     n::Integer,
#     τ::FT,
#     sum_Δxᵒ²::FT,
#     𝒫_D::InverseGamma{FT},
#     𝒫_x::MvNormal,
#     device::Device,
# ) where {FT<:AbstractFloat}
#     ~, B, N = size(xᵒ)
#     ϕ, ϕχτ4 = shape(𝒫_D), 4 * τ * scale(𝒫_D)
#     sum_Δxᵖ² =
#         sum_Δxᵒ² +
#         get_Δx²(view(xᵖ, :, :, n), view(xᵒ, :, :, n), view(xᵒ, :, :, get_index(n, N)))
#     Δln𝒫 = (-1.5 * B * (N - 1) - ϕ) * log((sum_Δxᵖ² + ϕχτ4) / (sum_Δxᵒ² + ϕχτ4))
#     n == 1 && (Δln𝒫 += get_Δln𝒫_x₁(view(xᵖ, :, :, 1), view(xᵒ, :, :, 1), 𝒫_x, device))
#     return Δln𝒫, sum_Δxᵖ²
# end

# function update_on_x!(
#     s::ChainStatus,
#     w::AbstractArray{Bool},
#     param::ExperimentalParameter,
#     device::Device,
# )
#     x = s.x
#     B = get_B(s)
#     xᵒ, Gᵒ, τ = view(s.x.value, :, 1:B, :), s.G, param.period
#     sum_Δxᵒ² = sum(diff(xᵒ, dims = 3) .^ 2)
#     xᵖ, Gᵖ = get_xᵖ(xᵒ, CuArray(diag(x.𝒬.Σ)), param)
#     diff_lnℒ = get_Δlnℒ_x(w, s.h.value, Gᵖ, Gᵒ, param.darkcounts, device) |> cpu
#     accepted = BitVector(undef, N)
#     for n in randperm(param.length)
#         ln𝓊 = log(rand())
#         ln𝓇, sum_Δxᵖ² = get_Δln𝒫_x(xᵖ, xᵒ, n, τ, sum_Δxᵒ², s.D.𝒫, x.𝒫, device)
#         ln𝓇 += diff_lnℒ[n]
#         accepted[n] = ln𝓇 > ln𝓊
#         if accepted[n]
#             xᵒ[:, :, n] .= xᵖ[:, :, n]
#             sum_Δxᵒ² = sum_Δxᵖ²
#             x.count[1] += 1
#         end
#         x.count[2] += 1
#     end
#     Gᵒ[:, :, accepted] .= Gᵖ[:, :, accepted]
# end

# get_ln𝓇_x(
#     w::AbstractArray{Bool,3},
#     G::AbstractArray{FT,3},
#     hᵖ::FT,
#     hᵒ::FT,
#     F::AbstractMatrix{FT},
#     𝒫::Gamma{FT},
# ) where {FT<:AbstractFloat} =
#     diff_lnℒ_h(w, G, hᵖ, hᵒ, F) + diff_ln𝒫_h(hᵖ, hᵒ, 𝒫) + diff_ln𝒬_h(hᵖ, hᵒ)