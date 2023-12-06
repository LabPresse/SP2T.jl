function add_lnℒ!(
    ln𝒫::AbstractVector{FT},
    𝐔::AbstractArray{FT,3},
    𝐖::AbstractArray{Bool,3},
    x::AbstractArray{FT,3},
    xᵖ::AbstractVector{FT},
    yᵖ::AbstractVector{FT},
    hτ::FT,
    PSF::AbstractPSF{FT},
    cpu::CPU,
) where {FT<:AbstractFloat}
    ln𝒫[1] += get_lnℒ(𝐖, 𝐔, cpu)
    for m = 1:size(x, 2)
        add_px_intensity!(𝐔, view(x, :, m:m, :), xᵖ, yᵖ, PSF, hτ)
        ln𝒫[m+1] += get_lnℒ(𝐖, 𝐔, cpu)
    end
    return ln𝒫
end

function add_lnℒ!(
    ln𝒫::AbstractVector{FT},
    𝐔::AbstractArray{FT,3},
    𝐖::AbstractArray{Bool,3},
    x::AbstractArray{FT,3},
    xᵖ::AbstractVector{FT},
    yᵖ::AbstractVector{FT},
    hτ::FT,
    PSF::AbstractPSF{FT},
    ::GPU,
) where {FT<:AbstractFloat}
    lnℒ = similar(𝐔)
    ln𝒫[1] += get_lnℒ!(lnℒ, 𝐖, 𝐔)
    for m = 1:size(x, 2)
        add_px_intensity!(𝐔, view(x, :, m:m, :), xᵖ, yᵖ, PSF, hτ)
        ln𝒫[m+1] += get_lnℒ!(lnℒ, 𝐖, 𝐔)
    end
    return ln𝒫
end

randc(ln𝒫::AbstractVector{FT}) where {FT<:AbstractFloat} =
    argmax(ln𝒫 .- log.(randexp(FT, length(ln𝒫))))

function sample_M(
    𝐖::AbstractArray{Bool,3},
    x::AbstractArray{FT,3},
    xᵖ::AbstractVector{FT},
    yᵖ::AbstractVector{FT},
    hτ::FT,
    𝐅::AbstractMatrix{FT},
    PSF::AbstractPSF{FT},
    𝒫::Geometric{FT},
    𝑇::FT,
    device::Device,
) where {FT<:AbstractFloat}
    ln𝒫 = collect(0:size(x, 2)) .* log(failprob(𝒫))
    # ln𝒫[1:end-1] .+= log1p(-q)
    ln𝒫 .*= 𝑇
    𝐔 = repeat(𝐅, 1, 1, size(x, 3))
    add_lnℒ!(ln𝒫, 𝐔, 𝐖, x, xᵖ, yᵖ, hτ, PSF, device)
    ln𝒫 ./= 𝑇
    return randc(ln𝒫) - 1
end

function shuffle!(x::AbstractArray, M::Integer)
    new_idx = vcat(randperm(M), M .+ randperm(size(x, 2) - M))
    @views x = x[:, new_idx, :]
    return x
end

function update_M!(s::ChainStatus, v::Video, device::Device)
    shuffle!(s.x.value, s.M.value)
    s.M.value = sample_M(
        v.data,
        s.x.value,
        v.param.pxboundsx,
        v.param.pxboundsy,
        s.h.value * v.param.period,
        v.param.darkcounts,
        v.param.PSF,
        s.M.𝒫,
        s.𝑇,
        device,
    )
    return s
end