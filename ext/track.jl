function SP2T.copyidxto!(xᵒ::CuArray, xᵖ::CuArray, idx::CuArray)
    xᵒ .= (idx .* xᵖ) .+ (.~idx .* xᵒ)
end