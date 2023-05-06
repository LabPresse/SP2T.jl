function (psf::CircularGaussianLorenzian)(
    x::AbstractVector{<:Real},
    pxboundsx::AbstractVector{<:Real},
    pxboundsy::AbstractVector{<:Real},
)
    σ_sqrt2 = psf.σ_ref_sqrt2 * sqrt(1 + (x[3] / psf.z_ref)^2)
    erfboundsx = (pxboundsx .- x[1]) ./ σ_sqrt2
    erfboundsy = (pxboundsy .- x[2]) ./ σ_sqrt2
    return @views erf.(erfboundsx[1:end-1], erfboundsx[2:end]) ./ 2 *
                  transpose(erf.(erfboundsy[1:end-1], erfboundsy[2:end]) ./ 2)
end

function (psf::CircularGaussianLorenzian)(
    x::AbstractMatrix{<:Real},
    pxboundsx::AbstractVector{<:Real},
    pxboundsy::AbstractVector{<:Real},
)
    g = zeros(length(pxboundsx) - 1, length(pxboundsy) - 1)
    for c in eachcol(x)
        g .+= psf(c, pxboundsx, pxboundsy)
    end
    return g
end

function (psf::CircularGaussianLorenzian)(
    x::AbstractArray{<:Real},
    pxboundsx::AbstractVector{<:Real},
    pxboundsy::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
)
    g = zeros(length(pxboundsx) - 1, length(pxboundsy) - 1)
    for (i, p) in enumerate(eachslice(x, dims = 3))
        g .+= w[i] .* psf(p, pxboundsx, pxboundsy)
    end
    return g
end

function add_sppsf!(
    g::AbstractMatrix{<:Real},
    erfboundsx::AbstractVector{<:Real},
    erfboundsy::AbstractVector{<:Real},
    w::Real,
)
    return mul!(
        g,
        (@views erf.(erfboundsx[1:end-1], erfboundsx[2:end])) ./ 2,
        (@views transpose(erf.(erfboundsy[1:end-1], erfboundsy[2:end]))) ./ 2,
        w,
        1,
    )
end

function add_psf!(
    g::AbstractMatrix{<:Real},
    erfboundsx::AbstractVector{<:Real},
    erfboundsy::AbstractVector{<:Real},
    x::AbstractVecOrMat{<:Real},
    pxboundsx::AbstractVector{<:Real},
    pxboundsy::AbstractVector{<:Real},
    w::Real,
    psf::CircularGaussianLorenzian,
)
    for c in eachcol(x)
        σ_sqrt2 = psf.σ_ref_sqrt2 * sqrt(1 + (c[3] / psf.z_ref)^2)
        erfboundsx .= (pxboundsx .- c[1]) ./ σ_sqrt2
        erfboundsy .= (pxboundsy .- c[2]) ./ σ_sqrt2
        add_sppsf!(g, erfboundsx, erfboundsy, w)
    end
    return g
end

function get_framepsf!(
    g::AbstractMatrix{<:Real},
    x::AbstractArray{<:Real},
    pxboundsx::AbstractVector{<:Real},
    pxboundsy::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
    psf::CircularGaussianLorenzian,
)
    erfboundsx = similar(pxboundsx, Float64)
    erfboundsy = similar(pxboundsy, Float64)
    for (i, p) in enumerate(eachslice(x, dims = 3))
        add_psf!(g, erfboundsx, erfboundsy, p, pxboundsx, pxboundsy, w[i], psf)
    end
    return g
end

function integrate_psf(
    x::AbstractArray{<:Real},
    params::ExperimentalParameters,
    w::AbstractVector{<:Real},
)
    g = zeros(params.pixelnumx, params.pixelnumy, params.length)
    K = length(w)
    for i = 1:params.length
        get_framepsf!(
            view(g, :, :, i),
            view(x, :, :, (i-1)*K+1:i*K),
            params.pixelboundsx,
            params.pixelboundsy,
            w,
            params.psf,
        )
    end
    g .*= params.exposure / (K - 1)
    return g
end

function get_readout(
    g::AbstractArray{<:Real},
    h::Real,
    F::Real,
    G::Real,
    params::ExperimentalParameters,
)
    return rand.(Gamma.((F * params.areatimesexposure .+ h .* g) ./ 2, 2 * G))
end
