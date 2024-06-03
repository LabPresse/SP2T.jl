SP2T.copyidxto!(x, xᵖ, idx::CuArray) = @. x = (idx * xᵖ) + (~idx * x)
