abstract type SimplifiedDistribution{T} end

struct Normal₃{T} <: SimplifiedDistribution{T}
    μ::T
    σ::T
end

_params(n::Normal₃) = n.μ, n.σ
struct BrownianTracks{A,V,B}
    value::A
    valueᵖ::A
    # Δx²::A
    # Δxᵖ²::A
    prior::Normal₃{V}
    perturbsize::V
    logratio::A
    logrands::A
    accepted::B
    counter::Matrix{Int}
end

function BrownianTracks(
    x::AbstractArray{T,3},
    xᵖ::AbstractArray{T,3},
    # Δx²::AbstractArray{T,3},
    prior::Normal₃{<:AbstractVector{T}},
    perturbsize::AbstractVector{T},
) where {T}
    logratio = similar(x, 1, 1, axes(x, 3))
    acceptance = fill!(similar(logratio, Bool), false)
    # @views Δxᵖ² = (xᵖ[:, :, 2:end] .- xᵖ[:, :, 1:end-1]) .^ 2
    return BrownianTracks(
        x,
        xᵖ,
        # Δx²,
        # Δxᵖ²,
        prior,
        perturbsize,
        logratio,
        similar(logratio),
        acceptance,
        zeros(Int, 2, 2),
    )
end

function BrownianTracks(; value, prior, perturbsize)
    valueᵖ = similar(value)
    # Δx² = similar(x, 3, size(x, 2), size(x, 3) - 1)
    # Δxᵖ² = similar(Δx²)
    logacceptance = similar(value, 1, 1, axes(value, 3))
    # acceptance = fill!(similar(logacceptance, Bool), false)
    acceptance = similar(logacceptance, Bool)
    # @views Δxᵖ² = (xᵖ[:, :, 2:end] .- xᵖ[:, :, 1:end-1]) .^ 2
    return BrownianTracks(
        value,
        valueᵖ,
        # Δx²,
        # Δxᵖ²,
        prior,
        perturbsize,
        logacceptance,
        similar(logacceptance),
        acceptance,
        zeros(Int, 2, 2),
    )
end

# maxnemitters(tracks::BrownianTracks) = size(tracks.x, 2)

# viewtracks(tracks::BrownianTracks, M) = view(tracks.x, :, 1:M, :)

ontracks(x::BrownianTracks, M) = view(x.value, :, 1:M, :)

# candidates(tracks::BrownianTracks, M) = view(tracks.xᵖ, :, 1:M, :)

# displacements(tracks::BrownianTracks, M) =
#     view(tracks.Δx², :, 1:M, :), view(tracks.Δxᵖ², :, 1:M, :)

logrand!(x::AbstractArray) = x .= log.(rand!(x))

# function logrand!(tracks::BrownianTracks)
#     rand!(tracks.lograndnums)
#     tracks.lograndnums .= log.(tracks.lograndnums)
#     return tracks
# end

# diff2!(Δx², x) = @views Δx² .= (x[:, :, 2:end] .- x[:, :, 1:end-1]) .^ 2

# function setΔx²!(tracks::BrownianTracks)
#     diff2!(tracks.Δx², tracks.x)
#     return tracks
# end

function simulate!(x, μ, σ, D)
    randn!(x)
    @views begin
        x[:, :, 1] .= x[:, :, 1] .* σ .+ μ
        x[:, :, 2:end] .*= √(2 * D)
    end
    return cumsum!(x, x, dims = 3)
end

# function simulate!(x, Δx², μ, σ, D)
#     simulate!(x, μ, σ, D)
#     @views Δx² .= x[:, :, 2:end] .^ 2
#     return x
# end

function MHinit!(x::BrownianTracks)
    fill!(x.logratio, -Inf)
    fill!(x.accepted, false)
    logrand!(x.logrands)
    return x
end

function propose!(xᵖ, x, σ)
    randn!(xᵖ)
    xᵖ .= xᵖ .* σ .+ x
end

# function propose!(tracks::BrownianTracks)
#     propose!(tracks.xᵖ, tracks.x, tracks.perturbsize)
#     return tracks
# end

# function propose!(tracks::BrownianTracks, M)
#     @views propose!(tracks.xᵖ[:, 1:M, :], tracks.x[:, 1:M, :], tracks.perturbsize)
#     return tracks
# end

# function set_ΔlogL!(ΔlogL, frames, 𝐔, 𝐔ᵖ, xᵖ, h, F, px, py, PSF, temperature, temp)
#     get_px_intensity!(𝐔ᵖ, xᵖ, h, F, px, py, PSF)
#     set_frame_Δlnℒ!(ΔlogL, frames, 𝐔, 𝐔ᵖ, temp)
#     return ΔlogL ./= temperature
# end

function add_Δlog𝒫!(ln𝓇, x₁, y₁, prior::Normal₃)
    ln𝓇 .+= sum(((x₁ .- prior.μ) .^ 2 - (y₁ .- prior.μ) .^ 2) ./ (2 .* prior.σ .^ 2))
    return ln𝓇
end

function add_Δlog𝒫!(x::BrownianTracks, M)
    @views add_Δlog𝒫!(x.logratio[:, :, 1], x.value[:, 1:M, 1], x.valueᵖ[:, 1:M, 1], x.prior)
    return x
end

# function add_Δln𝒫_x₁!(tracks::BrownianTracks, x, xᵖ)
#     add_Δln𝒫_x₁!(tracks.logacceptance, x, xᵖ, tracks.prior)
#     return tracks
# end

function copyidxto!(x, xᵖ, i)
    j = vec(i)
    @views x[:, :, j] .= xᵖ[:, :, j]
    return x
end

# copyidxto!(x, xᵖ, tracks::BrownianTracks) = copyidxto!(x, xᵖ, vec(tracks.accepted))

diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}) where {T} =
    @views Δx² .= (x[:, :, 2:end] .- x[:, :, 1:end-1]) .^ 2

function diff²!(
    Δx²::AbstractArray{T,3},
    x1::AbstractArray{T,3},
    x2::AbstractArray{T,3},
) where {T}
    @views begin
        Δx²[:, :, 1:2:end] .= (x1[:, :, 2:2:end] .- x2[:, :, 1:2:end-1]) .^ 2
        Δx²[:, :, 2:2:end] .= (x2[:, :, 3:2:end] .- x1[:, :, 2:2:end-1]) .^ 2
    end
    return Δx²
end

# function setΔx²!(tracks::BrownianTracks)
#     setΔx²!(tracks.Δx², tracks.x)
#     return tracks
# end

function add_ΔΔx²!(logr, Δx², Δy², idx1, idx2, D)
    @views begin
        logr[:, :, idx1] .+=
            sum(Δx²[:, :, idx1] .- Δy²[:, :, idx1], dims = (1, 2)) ./ (4 * D)
        logr[:, :, idx2.+1] .+=
            sum(Δx²[:, :, idx2] .- Δy²[:, :, idx2], dims = (1, 2)) ./ (4 * D)
    end
    return logr
end

add_odd_ΔΔx²!(logr, Δx², Δy², D) =
    add_ΔΔx²!(logr, Δx², Δy², 1:2:size(Δx², 3), 2:2:size(Δx², 3), D)

add_even_ΔΔx²!(logr, Δx², Δy², D) =
    add_ΔΔx²!(logr, Δx², Δy², 2:2:size(Δx², 3), 1:2:size(Δx², 3), D)

function update_counter!(x::BrownianTracks)
    @views x.counter[:, 2] .+= count(x.accepted), length(x.accepted)
    return x
end

# accept!(accepted::AbstractArray{Bool}, logr::AbstractArray, logu::AbstractArray) =
#     accepted .= logr .> logu

# function oddaccept!(tracks::BrownianTracks)
#     @views tracks.accepted[:, :, 1:2:end] .=
#         tracks.logacceptance[:, :, 1:2:end] .> tracks.lograndnums[:, :, 1:2:end]
#     return tracks
# end

# function evenaccept!(tracks::BrownianTracks)
#     @views tracks.accepted[:, :, 2:2:end] .=
#         tracks.logacceptance[:, :, 2:2:end] .> tracks.lograndnums[:, :, 2:2:end]
#     return tracks
# end

# function update_offtracks!(x::BrownianTracks, M::Integer, D::Real)
#     @views xᵒᶠᶠ = x.value[:, M+1:end, :]
#     μ, σ = _params(x.prior)
#     simulate!(xᵒᶠᶠ, μ, σ, D)
#     return x
# end