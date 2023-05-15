function (psf::CircularGaussianLorenzian)(
    x::AbstractVector{<:Real},
    px::AbstractVector{<:Real},
    py::AbstractVector{<:Real},
)
    σ_sqrt2 = psf.σ_ref_sqrt2 * sqrt(1 + (x[3] / psf.z_ref)^2)
    erfx = (px .- x[1]) ./ σ_sqrt2
    erfy = (py .- x[2]) ./ σ_sqrt2
    return @views erf.(erfx[1:end-1], erfx[2:end]) ./ 2 *
                  transpose(erf.(erfy[1:end-1], erfy[2:end]) ./ 2)
end

function (psf::CircularGaussianLorenzian)(
    x::AbstractMatrix{<:Real},
    px::AbstractVector{<:Real},
    py::AbstractVector{<:Real},
)
    g = zeros(length(px) - 1, length(py) - 1)
    for c in eachcol(x)
        g .+= psf(c, px, py)
    end
    return g
end

function (psf::CircularGaussianLorenzian)(
    x::AbstractArray{<:Real},
    px::AbstractVector{<:Real},
    py::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
)
    g = zeros(length(px) - 1, length(py) - 1)
    for (i, p) in enumerate(eachslice(x, dims = 3))
        g .+= w[i] .* psf(p, px, py)
    end
    return g
end

function add_sppsf!(
    g::AbstractMatrix{<:Real},
    erfx::AbstractVector{<:Real},
    erfy::AbstractVector{<:Real},
)
    return mul!(
        g,
        (@views erf.(erfx[1:end-1], erfx[2:end])) ./ 2,
        (@views transpose(erf.(erfy[1:end-1], erfy[2:end]))) ./ 2,
        1,
        1,
    )
end

function add_psf!(
    g::AbstractMatrix{<:Real},
    erfx::AbstractVector{<:Real},
    erfy::AbstractVector{<:Real},
    x::AbstractVecOrMat{<:Real},
    px::AbstractVector{<:Real},
    py::AbstractVector{<:Real},
    psf::CircularGaussianLorenzian,
)
    for c in eachcol(x)
        σ_sqrt2 = psf.σ_ref_sqrt2 * sqrt(1 + (c[3] / psf.z_ref)^2)
        erfx .= (px .- c[1]) ./ σ_sqrt2
        erfy .= (py .- c[2]) ./ σ_sqrt2
        add_sppsf!(g, erfx, erfy)
    end
    return g
end

function get_framepsf!(
    g::AbstractMatrix{<:Real},
    x::AbstractVecOrMat{<:Real},
    px::AbstractVector{<:Real},
    py::AbstractVector{<:Real},
    psf::CircularGaussianLorenzian,
)
    erfx = similar(px, Float64)
    erfy = similar(py, Float64)
    add_psf!(g, erfx, erfy, x, px, py, psf)
    return g
end

function integrate_psf(
    x::AbstractArray{<:Real},
    params::ExperimentalParameters,
)
    g = zeros(params.pixelnumx, params.pixelnumy, params.length)
    for i = 1:params.length
        get_framepsf!(
            view(g, :, :, i),
            view(x, :, :, i),
            params.pixelboundsx,
            params.pixelboundsy,
            params.psf,
        )
    end
    g .*= params.exposure
    return g
end

function get_readout(
    g::AbstractArray{<:Real},
    h::Real,
    F::Real,
    params::ExperimentalParameters,
)
    u = F * params.areatimesexposure .+ h .* g
    p = -expm1.(-u)
    return rand(eltype(p), size(p)) .< p .&& params.validpixel
end
