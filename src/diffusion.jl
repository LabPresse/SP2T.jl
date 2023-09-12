function sample_D(
    x::AbstractArray{<:AbstractFloat,3},
    𝒫::InverseGamma,
    T::Real,
    𝑇::Real = 1.0,
)
    Δx² = diff(x, dims = 3) .^ 2
    Δshape = length(Δx²) / 2
    Δscale = sum(Δx²) / (4 * T)
    newparams = params(𝒫) .+ (Δshape, Δscale) ./ 𝑇
    return rand(InverseGamma(newparams...))
end

function update_D!(s::ChainStatus, param::ExperimentalParameter)
    s.D.value = sample_D(view_on_x(s), s.D.𝒫, param.period, s.𝑇)
    return s
end
