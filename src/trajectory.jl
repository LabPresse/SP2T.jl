#* Forward functions
function setinitx!(x::AbstractArray{FT}, 𝒫::Distribution, ::CPU) where {FT<:AbstractFloat}
    x .= rand(𝒫, size(x, 2))
    return x
end

function setinitx!(x::AbstractArray{FT}, 𝒫::Distribution, ::GPU) where {FT<:AbstractFloat}
    x .= CuArray(rand(𝒫, size(x, 2)))
    return x
end

function setΔx!(x::AbstractArray{FT,3}, σ::FT, ::CPU) where {FT<:AbstractFloat}
    x .= randn(FT, size(x)...)
    x .*= σ
    return x
end

function setΔx!(x::AbstractArray{FT,3}, σ::FT, ::GPU) where {FT<:AbstractFloat}
    CUDA.randn!(x)
    x .*= σ
    return x
end

function simulate!(
    x::AbstractArray{FT,3},
    𝒫::Distribution,
    D::FT,
    τ::FT,
    device::Device,
) where {FT<:AbstractFloat}
    @views begin
        setinitx!(x[:, :, 1], 𝒫, device)
        setΔx!(x[:, :, 2:end], √(2 * D * τ), device)
    end
    cumsum!(x, x, dims = 3)
    return x
end

#* Inverse functions
propose_x(xᵒ::AbstractArray{FT,3}, 𝒬::MvNormal, ::CPU) where {FT<:AbstractFloat} =
    xᵒ .+ sqrt.(viewdiag(𝒬.Σ)) .* randn(size(xᵒ)...)

propose_x(xᵒ::AbstractArray{FT,3}, 𝒬::MvNormal, ::GPU) where {FT<:AbstractFloat} =
    xᵒ .+ sqrt.(CuArray(diag(𝒬.Σ))) .* CUDA.randn(size(xᵒ)...)

# get_ΔΔx²(
#     xᵒ::AbstractMatrix{FT},
#     xᵖ::AbstractMatrix{FT},
#     neighbourx::AbstractArray{FT},
# ) where {FT<:AbstractFloat} = sum((xᵒ .- neighbourx) .^ 2 .- (xᵖ .- neighbourx) .^ 2)

# function get_Δlnℒ_x(
#     w::AbstractArray{Bool,3},
#     Gᵖ::AbstractArray{FT,3},
#     Gᵒ::AbstractArray{FT,3},
#     hτ::FT,
#     F::AbstractMatrix{FT},
# ) where {FT<:AbstractFloat}
#     uᵖ = F .+ hτ .* Gᵖ
#     uᵒ = F .+ hτ .* Gᵒ
#     Δlnℒ = uᵒ .- uᵖ
#     Δlnℒ[w] .+= logexpm1.(uᵖ[w]) .- logexpm1.(uᵒ[w])
#     return sum(Δlnℒ, dims = (1, 2))
# end

neighbour_indices(n::Integer, N::Integer) = ifelse(n == 1, 2, n-1:2:min(n + 1, N))

view_neighbour(x::AbstractArray{<:AbstractFloat,3}, n::Integer, N::Integer) =
    view(x, :, :, neighbour_indices(n, N))

function add_Δln𝒫_x₁!(
    ln𝓇::AbstractVector{FT},
    xᵒ::AbstractMatrix{FT},
    xᵖ::AbstractMatrix{FT},
    𝒫::MvNormal,
) where {FT<:AbstractFloat}
    ln𝓇[1] += sum(((xᵒ .- 𝒫.μ) .^ 2 - (xᵖ .- 𝒫.μ) .^ 2) ./ (2 .* viewdiag(𝒫.Σ)))
    return ln𝓇
end

function add_Δln𝒫_x₁!(
    ln𝓇::AbstractArray{FT,3},
    xᵒ_cu::AbstractMatrix{FT},
    xᵖ_cu::AbstractMatrix{FT},
    𝒫::MvNormal,
) where {FT<:AbstractFloat}
    xᵒ = Array(xᵒ_cu)
    xᵖ = Array(xᵖ_cu)
    CUDA.@allowscalar ln𝓇[1] +=
        sum(((xᵒ .- 𝒫.μ) .^ 2 - (xᵖ .- 𝒫.μ) .^ 2) ./ (2 .* viewdiag(𝒫.Σ)))
    return ln𝓇
end

# get_Δln𝒫_x(
#     xᵒ::AbstractMatrix{FT},
#     xᵖ::AbstractMatrix{FT},
#     xᶜ::AbstractArray{FT},
#     fourDτ::FT,
# ) where {FT<:AbstractFloat} = get_ΔΔx²(xᵒ, xᵖ, xᶜ) / fourDτ

function get_acceptance!(
    xᵒ::AbstractArray{FT,3},
    xᵖ::AbstractArray{FT,3},
    ln𝓇::AbstractVector{FT},
    fourDτ::FT,
) where {FT<:AbstractFloat}
    N = size(ln𝓇, 3)
    accepted = BitVector(undef, N)
    @inbounds for n in randperm(N)
        xᵒₙ, xᵖₙ, xⁿₙ = view(xᵒ, :, :, n), view(xᵖ, :, :, n), view_neighbour(xᵒ, n, N)
        ln𝓇[n] += sum((xᵒₙ .- xⁿₙ) .^ 2 .- (xᵖₙ .- xⁿₙ) .^ 2) / fourDτ
        accepted[n] = ln𝓇[n] > log(rand(FT))
        accepted[n] && (xᵒₙ .= xᵖₙ)
    end
    return accepted
end

function copyidxto!(
    xᵒ::AbstractArray{FT,3},
    xᵖ::AbstractArray{FT,3},
    idx::AbstractVector{Bool},
) where {FT<:AbstractFloat}
    @views xᵒ[:, :, idx] .= xᵖ[:, :, idx]
end

function copyidxto!(
    xᵒ::AbstractArray{FT,N},
    xᵖ::AbstractArray{FT,N},
    idx::AbstractArray{Bool,N},
) where {FT<:AbstractFloat,N}
    xᵒ .= (idx .* xᵖ) .+ (.~idx .* xᵒ)
end

getindices(N::Integer) = ifelse(isodd(N), (1:2:N-2, 2:2:N-1), (1:2:N-1, 2:2:N-2))

function set_Δxᵒ²!(Δxᵒ²::AbstractArray{FT}, xᵒ::AbstractArray{FT}) where {FT<:AbstractFloat}
    @views Δxᵒ² .= (xᵒ[:, :, 2:end] .- xᵒ[:, :, 1:end-1]) .^ 2
    return Δxᵒ²
end

function set_Δxᵖ²!(
    Δxᵖ²::AbstractArray{FT,3},
    xᵒ::AbstractArray{FT,3},
    xᵖ::AbstractArray{FT,3},
    firstidx::Integer,
) where {FT<:AbstractFloat}
    x1, x2 = ifelse(isone(firstidx), (xᵒ, xᵖ), (xᵖ, xᵒ))
    @views begin
        Δxᵖ²[:, :, 1:2:end] .= x1[:, :, 2:2:end] .- x2[:, :, 1:2:end-1]
        Δxᵖ²[:, :, 2:2:end] .= x2[:, :, 3:2:end] .- x1[:, :, 2:2:end-1]
    end
    Δxᵖ² .^= 2
    return Δxᵖ²
end

function add_ΔΔx²!(
    ln𝓇::AbstractArray{FT,3},
    Δxᵒ²::AbstractArray{FT,3},
    Δxᵖ²::AbstractArray{FT,3},
    idx1::AbstractRange,
    idx2::AbstractRange,
    fourDτ::FT,
) where {FT<:AbstractFloat}
    @views begin
        ln𝓇[:, :, idx1] .+=
            sum(Δxᵒ²[:, :, idx1] .- Δxᵖ²[:, :, idx1], dims = (1, 2)) ./ fourDτ
        ln𝓇[:, :, idx2.+1] .+=
            sum(Δxᵒ²[:, :, idx2] .- Δxᵖ²[:, :, idx2], dims = (1, 2)) ./ fourDτ
    end
    return ln𝓇
end

function get_acceptance!(
    xᵒ::AbstractArray{FT,3},
    xᵖ::AbstractArray{FT,3},
    ln𝓇::AbstractArray{FT,3},
    fourDτ::FT,
) where {FT<:AbstractFloat}
    N = size(ln𝓇, 3)
    idx1, idx2 = getindices(N)
    Δxᵒ² = similar(xᵒ, size(xᵒ, 1), size(xᵒ, 2), N - 1)
    Δxᵖ² = similar(Δxᵒ²)
    accepted = CUDA.zeros(Bool, 1, 1, N)
    for i = 1:2
        set_Δxᵒ²!(Δxᵒ², xᵒ)
        set_Δxᵖ²!(Δxᵖ², xᵒ, xᵖ, i)
        add_ΔΔx²!(ln𝓇, Δxᵒ², Δxᵖ², idx1, idx2, fourDτ)
        @views accepted[:, :, i:2:end] .=
            ln𝓇[:, :, i:2:end] .> log.(CUDA.rand(FT, 1, 1, length(i:2:N)))
        copyidxto!(xᵒ, xᵖ, accepted)
        idx1, idx2 = idx2, idx1
    end
    return accepted
end

function update_on_x!(
    s::ChainStatus,
    𝐖::AbstractArray{Bool,3},
    param::ExperimentalParameter,
    device::CPU,
)
    xᵒ, 𝐔ᵒ = view_on_x(s), s.𝐔
    xᵖ = propose_x(xᵒ, s.x.𝒬, device)
    𝐔ᵖ = get_px_intensity(
        xᵖ,
        param.pxboundsx,
        param.pxboundsy,
        s.h.value * param.period,
        param.darkcounts,
        param.PSF,
    )
    ln𝓇 = get_frame_Δlnℒ(𝐖, 𝐔ᵒ, 𝐔ᵖ, device)
    ln𝓇[1] += add_Δln𝒫_x₁!(ln𝓇, view(xᵖ, :, :, 1), view(xᵒ, :, :, 1), s.x.prior)
    accepted = get_acceptance!(xᵒ, xᵖ, ln𝓇, 4 * s.D.value * param.period)
    s.x.counter[:, 2] .+= count(accepted), length(accepted)
    copyidxto!(𝐔ᵒ, 𝐔ᵖ, accepted)
    return s
end

function update_on_x!(
    s::ChainStatus,
    𝐖::AbstractArray{Bool,3},
    param::ExperimentalParameter,
    device::GPU,
)
    xᵒ, 𝐔ᵒ = view_on_x(s), s.𝐔
    xᵖ = propose_x(xᵒ, s.x.𝒬, device)
    𝐔ᵖ = get_px_intensity(
        xᵖ,
        param.pxboundsx,
        param.pxboundsy,
        s.h.value * param.period,
        param.darkcounts,
        param.PSF,
    )
    ln𝓇 = get_frame_Δlnℒ(𝐖, 𝐔ᵒ, 𝐔ᵖ, device)
    add_Δln𝒫_x₁!(ln𝓇, view(xᵖ, :, :, 1), view(xᵒ, :, :, 1), s.x.prior)
    accepted = get_acceptance!(xᵒ, xᵖ, ln𝓇, 4 * s.D.value * param.period)
    s.x.counter[:, 2] .+= count(accepted), length(accepted)
    copyidxto!(𝐔ᵒ, 𝐔ᵖ, accepted)
    return s
end

update_off_x!(s::ChainStatus, param::ExperimentalParameter, device::Device) =
    simulate!(view_off_x(s), s.x.prior, s.D.value, param.period, device)

function update_x!(s::ChainStatus, v::Video, device::Device)
    update_off_x!(s, v.param, device)
    update_on_x!(s, v.frames, v.param, device)
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
#     for n in randperm(size(w, 3))
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