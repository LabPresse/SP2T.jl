unionalltypeof(::Gamma) = Gamma
unionalltypeof(::InverseGamma) = InverseGamma
unionalltypeof(::Matrix) = Matrix
unionalltypeof(::Vector) = Vector
unionalltypeof(::Array) = Array

anneal(logℒ::T, 𝑇::T) where {T} = logℒ / 𝑇
anneal!(logℒ::AbstractVector{T}, 𝑇::T) where {T} = logℒ ./= 𝑇