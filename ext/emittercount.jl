function SP2T.add_lnℒ!(
    ln𝒫::CuVector{T},
    𝐔::CuArray{T,3},
    𝐖::CuArray{Bool,3},
    x::CuArray{T,3},
    xᵖ::CuVector{T},
    yᵖ::CuVector{T},
    hτ::T,
    PSF::AbstractPSF{T},
) where {T<:AbstractFloat}
    lnℒ = similar(𝐔)
    ln𝒫[1] += get_lnℒ!(lnℒ, 𝐖, 𝐔)
    for m = 1:size(x, 2)
        add_px_intensity!(𝐔, view(x, :, m:m, :), xᵖ, yᵖ, PSF, hτ)
        ln𝒫[m+1] += get_lnℒ!(lnℒ, 𝐖, 𝐔)
    end
    return ln𝒫
end

function SP2T.shuffle_on_x!(x::CuArray, B)
    x[:, 1:B, :] = view(x, :, randperm(B), :)
    return x
end