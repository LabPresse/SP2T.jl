SP2T.Sample(
    x::CuArray{T,3},
    M::Integer,
    D::T,
    h::T,
    i::Integer,
    𝑇::T,
    log𝒫::T,
    logℒ::T,
) where {T} = Sample(Array(view(x, :, :, 1:M)), D, h, i, 𝑇, log𝒫, logℒ)