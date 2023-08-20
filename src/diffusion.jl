function sample_D(
    x::AbstractArray{<:AbstractFloat,3},
    𝑃::InverseGamma,
    T::Real,
    𝕋::Real = 1.0,
)
    Δx² = diff(x, dims = 3) .^ 2
    Δshape = length(Δx²) / 2
    Δscale = sum(Δx²) / (4 * T)
    newparams = params(𝑃) .+ (Δshape, Δscale) ./ 𝕋
    ℙ = InverseGamma(newparams...)
    return rand(ℙ)
end

function update_D!(s::FullSample, prior::Sampleable, param::ExperimentalParameter)
    s.D = sample_D(view_x(s), prior, param.period)
    return s
end
