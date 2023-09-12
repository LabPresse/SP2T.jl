function get_log𝕃(
    w::AbstractArray{Bool,3},
    G::AbstractArray{<:AbstractFloat,3},
    h::AbstractFloat,
    F::AbstractMatrix{<:AbstractFloat};
    onGPU::Bool,
)
    u = F .+ h .* G
    if onGPU
        return sum(w .* logexpm1.(u) .- u)
    else
        return sum(logexpm1.(u[w])) - sum(u)
    end
end