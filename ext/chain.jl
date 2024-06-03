SP2T.Sample(x::CuArray, M, D, h, i, T, log𝒫, logℒ) =
    Sample(Array(view(x, :, 1:M, :)), D, h, i, T, log𝒫, logℒ)