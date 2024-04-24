mutable struct EmitterCount{T<:AbstractFloat,VT<:AbstractVector{T}}
    value::Int
    prior::Geometric{T}
    logposterior::VT
end

getmaxcount(e::EmitterCount) = length(e.logposterior)

function addlogprior!(e::EmitterCount)
    e.logposterior .= (0:length(e.logposterior)-1) .* log(failprob(e.prior))
    return e
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

randc(logp::AbstractVector{T}) where {T} = argmax(logp .- log.(randexp!(similar(logp))))

function sample_M(
    𝐖::AbstractArray{Bool,3},
    x::AbstractArray{T,3},
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    hτ::T,
    𝐅::AbstractMatrix{T},
    PSF::AbstractPSF{T},
    prior::Geometric{T},
    temperature::T,
    device::Device,
) where {T}
    ln𝒫 = collect(0:size(x, 2)) .* log(failprob(prior))
    # ln𝒫[1:end-1] .+= log1p(-q)
    ln𝒫 .*= temperature
    𝐔 = repeat(𝐅, 1, 1, size(x, 3))
    add_lnℒ!(ln𝒫, 𝐔, 𝐖, x, xᵖ, yᵖ, hτ, PSF, device)
    ln𝒫 ./= temperature
    return randc(ln𝒫) - 1
end

function shuffle_on_x!(x::AbstractArray{<:Real,3}, B::Integer, ::CPU)
    x[:, 1:B, :] = x[:, randperm(B), :]
    return x
end

function shuffle_on_x!(x::AbstractArray{<:Real,3}, B::Integer, ::GPU)
    x[:, 1:B, :] = view(x, :, randperm(B), :)
    return x
end

function update_emittercount!(s::ChainStatus, v::Video, device::Device)
    shuffle_on_x!(s.tracks.value, s.emittercount.value, device)
    s.emittercount.value = sample_M(
        v.frames,
        s.tracks.value,
        v.param.pxboundsx,
        v.param.pxboundsy,
        s.brightness.value * v.param.period,
        v.param.darkcounts,
        v.param.PSF,
        s.emittercount.prior,
        s.temperature,
        device,
    )
    return s
end