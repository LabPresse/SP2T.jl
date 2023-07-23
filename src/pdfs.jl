# This file contains UNNORMALIZED probability density functions.
# The goal is to make things simple and fast. 
# For normalized pdfs use "Distributions.jl".

"""
    logbetapdf(x, α, β)

    The log pdf of the Beta distribution with shape parameters `α` and `β`.
"""
logbetapdf(x, α, β) = (α - 1) * log(x) + (β - 1) * log1p(-x)

"""
    loggampdf(x, α, β)

The log pdf of the Gamma distribution with shape `α` and scale `β`.
"""
loggampdf(x::Real, α, β) = (α - 1) * x - x / β
loggampdf(x::AbstractVector{Float64}, α, β) = sum((α - 1) .* x - x ./ β)

"""
    loginvgampdf(x, α, β)

    The log pdf of the Inverse-Gamma distribution with shape `α` and scale `β`.
"""
loginvgampdf(x::Real, α, β) = (-α - 1) * x - β / x
loginvgampdf(x::AbstractVector{Float64}, α, β) = sum((-α - 1) .* x - β ./ x)

"""
    logdirpdf(x, α)

    The log pdf of the Dirichlet distribution with concentration parameter `α`.
"""
logdirpdf(x, α) = sum((α .- 1) .* log.(x))

"""
    logbrownpdf(x, D, t)

    The log pdf of a n-D Brownian motion trajectory (`x`) with diffusion coefficient `D`. As 'D' is often inferred, the D-dependence in the normalization factor is not dropped. The number of dimemsions, n, is `x`'s number of columns.
"""
function logbrownpdf(x::AbstractMatrix{Float64}, D::Real, t::AbstractVector{Float64})
    Δx² = sum(diff(x, dims = 1) .^ 2, dims = 2)
    twoσ² = 2 * D * diff(t)
    return -log(D) / 2 * length(Δx²) * size(x, 2) - sum(Δx² ./ twoσ²)
end

"""
    logcatpdf(x, p)

    The log pdf of a Categorical distribution with probability vector `p`. `p` does not need to be normalized.
"""
logcatpdf(x::Integer, p) = log(p[x])
logcatpdf(x::Union{AbstractVector{Int8},AbstractVector{Int}}, p) = sum(log.(p[x]))

"""
    logbernpdf(c, p)

    The log pdf of a Bernoulli distribution with success probability `p`. `p` should be normalized.
"""
logbernpdf(x::Bool, p) = x ? log(p) : log1p(-p)
function logbernpdf(x::AbstractVector{Bool}, p)
    n = count(x)
    return n * log(p) + (length(x) - n) * log1p(-p::Real)
end