function copyidxto!(xᵒ::CuArray{T,N}, xᵖ::CuArray{T,N}, idx::CuArray{Bool,N}) where {T,N}
    xᵒ .= (idx .* xᵖ) .+ (.~idx .* xᵒ)
end