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

get_Δx²(
    xᵖ::AbstractMatrix{FT},
    xᵒ::AbstractMatrix{FT},
    neighbourx::AbstractArray{FT},
) where {FT<:AbstractFloat} = sum((xᵖ .- neighbourx) .^ 2 .- (xᵒ .- neighbourx) .^ 2)

function get_Δlnℒ_x(
    w::AbstractArray{Bool,3},
    h::AbstractFloat,
    Gᵖ::AbstractArray{FT,3},
    Gᵒ::AbstractArray{FT,3},
    F::AbstractMatrix{FT},
    ::GPU,
) where {FT<:AbstractFloat}
    uᵖ = F .+ h .* Gᵖ
    uᵒ = F .+ h .* Gᵒ
    Δlnℒ = w .* (logexpm1.(uᵖ) .- logexpm1.(uᵒ)) .- (uᵖ .- uᵒ)
    return Array(sum(Δlnℒ, dims = (1, 2)))
end

function get_Δlnℒ_x(
    w::AbstractArray{Bool,3},
    h::AbstractFloat,
    Gᵖ::AbstractArray{FT,3},
    Gᵒ::AbstractArray{FT,3},
    F::AbstractMatrix{FT},
    ::CPU,
) where {FT<:AbstractFloat}
    uᵖ = F .+ h .* Gᵖ
    uᵒ = F .+ h .* Gᵒ
    Δlnℒ = (logexpm1.(uᵖ[w]) .- logexpm1.(uᵒ[w])) .- (uᵖ .- uᵒ)
    return sum(Δlnℒ, dims = (1, 2))
end

neighbour_indices(n::Integer, N::Integer) = n == 1 ? 2 : n-1:2:min(n + 1, N)

neighbourx(x::AbstractArray{<:AbstractFloat,3}, n::Integer, N::Integer) =
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
    ΔΔx² = get_Δx²(xᵖ, xᵒ, xᶜ)
    Δln𝒫 = -ΔΔx² / fourDτ
    isfirstframe && (Δln𝒫 += get_Δln𝒫_x₁(xᵖ, xᵒ, 𝒫, device))
    return Δln𝒫
end

function get_Δln𝒫_x(
    xᵖ::AbstractArray{FT,3},
    xᵒ::AbstractArray{FT,3},
    n::Integer,
    τ::FT,
    D::FT,
    𝒫::MvNormal,
    device::Device,
) where {FT<:AbstractFloat}
    N = size(xᵒ, 3)
    ΔΔx² = get_Δx²(view(xᵖ, :, :, n), view(xᵒ, :, :, n), neighbourx(xᵒ, n, N))
    Δln𝒫 = -ΔΔx² / (4 * D * τ)
    n == 1 && (Δln𝒫 += get_Δln𝒫_x₁(view(xᵖ, :, :, 1), view(xᵒ, :, :, 1), 𝒫, device))
    return Δln𝒫
end

function update_on_x!(
    s::ChainStatus,
    w::AbstractArray{Bool},
    param::ExperimentalParameter,
    device::Device,
)
    N, F = param.length, param.darkcounts
    h, fourDτ = s.h.value, 4 * s.D.value * param.period
    𝒫, 𝒬, count = s.x.𝒫, s.x.𝒬, s.x.count
    xᵒ, Gᵒ = view_on_x(s), s.G
    xᵖ, Gᵖ = get_xᵖ(xᵒ, 𝒬, param, device)
    Δlnℒ = get_Δlnℒ_x(w, h, Gᵖ, Gᵒ, F, device)
    for n in randperm(N)
        ln𝓊 = log(rand())
        xᵖₙ, xᵒₙ, xᶜₙ = view(xᵖ, :, :, n), view(xᵒ, :, :, n), neighbourx(xᵒ, n, N)
        ln𝓇 = Δlnℒ[n] + get_Δln𝒫_x(xᵖₙ, xᵒₙ, xᶜₙ, fourDτ, n == 1, 𝒫, device)
        if ln𝓇 > ln𝓊
            xᵒ[:, :, n] .= xᵖ[:, :, n]
            count[1] += 1
        end
        count[2] += 1
    end
    #? potential improvemnt
    s.G = get_pxPSF(xᵒ, param.pxboundsx, param.pxboundsy, param.PSF)
end

update_off_x!(s::ChainStatus, param::ExperimentalParameter, device::GPU) =
    simulate!(view_off_x(s), s.x.𝒫, s.D.value, param.period, device::GPU)

update_off_x!(s::ChainStatus, param::ExperimentalParameter, device::CPU) =
    simulate!(view_off_x(s), s.x.𝒫, s.D.value, param.period, device::CPU)

function update_x!(
    s::ChainStatus,
    w::AbstractArray{Bool},
    param::ExperimentalParameter,
    device::GPU,
)
    update_off_x!(s::ChainStatus, param::ExperimentalParameter, device::GPU)
    update_on_x!(
        s::ChainStatus,
        w::AbstractArray{Bool},
        param::ExperimentalParameter,
        device::GPU,
    )
    return s
end

function update_x!(
    s::ChainStatus,
    w::AbstractArray{Bool},
    param::ExperimentalParameter,
    device::CPU,
)
    update_off_x!(s::ChainStatus, param::ExperimentalParameter, device::CPU)
    update_on_x!(
        s::ChainStatus,
        w::AbstractArray{Bool},
        param::ExperimentalParameter,
        device::CPU,
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
#     for n in randperm(param.length)
#         ln𝓊 = log(rand())
#         ln𝓇, sum_Δxᵖ² = get_Δln𝒫_x(xᵖ, xᵒ, n, τ, sum_Δxᵒ², s.D.𝒫, x.𝒫, device)
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

# get_ln𝓇_x(
#     w::AbstractArray{Bool,3},
#     G::AbstractArray{FT,3},
#     hᵖ::FT,
#     hᵒ::FT,
#     F::AbstractMatrix{FT},
#     𝒫::Gamma{FT},
# ) where {FT<:AbstractFloat} =
#     diff_lnℒ_h(w, G, hᵖ, hᵒ, F) + diff_ln𝒫_h(hᵖ, hᵒ, 𝒫) + diff_ln𝒬_h(hᵖ, hᵒ)