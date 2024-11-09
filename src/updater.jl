function update_odd!(
    𝐱::AbstractArray{T,3},
    𝐲::AbstractArray{T,3},
    Δ𝐱²::AbstractArray{T,3},
    Δ𝐲²::AbstractArray{T,3},
    msd::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΔΔx²::AbstractArray{T,3},
    ΣΔΔ𝐱²::AbstractVector{T},
    Δlogℒ::AbstractVector{T},
) where {T}
    oddΔlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², msd, ΔΔx², ΣΔΔ𝐱²)
    @views begin
        logr[1:2:end] .+= Δlogℒ[1:2:end]
        accept[1:2:end] .= logr[1:2:end] .> 0
    end
    copyidxto!(𝐱, 𝐲, accept)
end

function update_even!(
    𝐱::AbstractArray{T,3},
    𝐲::AbstractArray{T,3},
    Δ𝐱²::AbstractArray{T,3},
    Δ𝐲²::AbstractArray{T,3},
    msd::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΔΔx²::AbstractArray{T,3},
    ΣΔΔ𝐱²::AbstractVector{T},
    Δlogℒ::AbstractVector{T},
) where {T}
    evenΔlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², msd, ΔΔx², ΣΔΔ𝐱²)
    @views begin
        logr[2:2:end] .+= Δlogℒ[2:2:end]
        accept[2:2:end] .= logr[2:2:end] .> 0
    end
    copyidxto!(𝐱, 𝐲, accept)
end

function update_odd_even!(
    𝐱::AbstractArray{T,3},
    𝐲::AbstractArray{T,3},
    Δ𝐱²::AbstractArray{T,3},
    Δ𝐲²::AbstractArray{T,3},
    msd::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΔΔx²::AbstractArray{T,3},
    ΣΔΔ𝐱²::AbstractVector{T},
    Δlogℒ::AbstractVector{T},
) where {T}
    update_odd!(𝐱, 𝐲, Δ𝐱², Δ𝐲², msd, logr, accept, ΔΔx², ΣΔΔ𝐱², Δlogℒ)
    update_even!(𝐱, 𝐲, Δ𝐱², Δ𝐲², msd, logr, accept, ΔΔx², ΣΔΔ𝐱², Δlogℒ)
    return 𝐱
end

function update_ontracks!(
    tracks::Tracks{T},
    nemittersᵥ::Integer,
    msdᵥ::T,
    brightnessᵥ::T,
    measurements::AbstractArray{<:Union{T,UInt16}},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    MHinit!(tracks)
    x, y, Δx², Δy², ΔΔx² = trackviews(tracks, nemittersᵥ)
    propose!(y, x, tracks.perturbsize)
    pxcounts!(detector, x, y, brightnessᵥ, psf)
    Δlogℒ!(detector, measurements)
    tracks.logratio .+= anneal!(detector.framelogℒ, 𝑇)
    addΔlogπ₁!(tracks.logratio, x, y, tracks.prior)
    update_odd_even!(
        x,
        y,
        Δx²,
        Δy²,
        msdᵥ,
        tracks.logratio,
        tracks.accepted,
        ΔΔx²,
        tracks.ΣΔdisplacement²,
        detector.framelogℒ,
    )
    counter!(tracks)
    return tracks
end

function update_offtracks!(tracks::Tracks{T}, nemittersᵥ::Integer, msdᵥ::T) where {T}
    @views x = tracks.value[:, :, nemittersᵥ+1:end]
    μ, σ = params(tracks.prior)
    simulate!(x, μ, σ, msdᵥ)
    return tracks
end

function update!(
    msd::MeanSquaredDisplacement{T},
    trackᵥ::AbstractArray{T,3},
    displacements::AbstractArray{T,3},
    𝑇::T,
) where {T}
    diff²!(displacements, trackᵥ)
    return sample!(msd, displacements, 𝑇)
end

function update!(
    nemitters::NTracks{T},
    trackᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    setlogℒ!(nemitters, trackᵥ, brightnessᵥ, measurements, detector, psf)
    setlog𝒫!(nemitters, 𝑇)
    sample!(nemitters)
    return nemitters
end

function parametricMCMC!(
    tracks::Tracks{T},
    nemitters::NTracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    update_ontracks!(
        tracks,
        nemitters.value,
        msd.value,
        brightness.value,
        measurements,
        detector,
        psf,
        𝑇,
    )
    update!(
        msd,
        view(tracks.value, :, :, 1:nemitters.value),
        view(tracks.displacement₁², :, :, 1:nemitters.value),
        𝑇,
    )
    return tracks, msd
end

function nonparametricMCMC!(
    tracks::Tracks{T},
    nemitters::NTracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    update_offtracks!(tracks, nemitters.value, msd.value)
    if any(nemitters)
        update_ontracks!(
            tracks,
            nemitters.value,
            msd.value,
            brightness.value,
            measurements,
            detector,
            psf,
            𝑇,
        )
        shuffleactive!(tracks, nemitters)
    end
    update!(msd, tracks.value, tracks.displacement₁², 𝑇)
    update!(nemitters, tracks.value, brightness.value, measurements, detector, psf, 𝑇)
    return tracks, msd, nemitters
end

function runMCMC!(
    chain::Chain{T},
    tracks::Tracks{T},
    nemitters::NTracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    niters::Integer,
    parametric::Bool,
) where {T}
    prev_niters = chain.samples[end].iteration
    initintensity!(detector)
    nemitters.logℒ[1] = logℒ!(detector, measurements)
    nextsample! = parametric ? parametricMCMC! : nonparametricMCMC!
    @showprogress 1 "Computing..." for iter in prev_niters .+ (1:niters)
        𝑇 = temperature(chain, iter)
        nextsample!(tracks, nemitters, msd, brightness, measurements, detector, psf, 𝑇)
        extend!(
            chain,
            tracks,
            nemitters,
            msd,
            brightness,
            measurements,
            detector,
            psf,
            iter,
            𝑇,
        )
    end
end

function runMCMC(;
    tracks::Tracks{T},
    nemitters::NTracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    niters::Integer = 1000,
    sizelimit::Integer = 1000,
    annealing::Union{AbstractAnnealing{T},Nothing} = nothing,
    parametric::Bool = false,
) where {T}
    isnothing(annealing) && (annealing = ConstantAnnealing{T}(1))
    chain = Chain([Sample(tracks, nemitters, msd, brightness)], sizelimit, annealing)
    runMCMC!(
        chain,
        tracks,
        nemitters,
        msd,
        brightness,
        measurements,
        detector,
        psf,
        niters,
        parametric,
    )
    return chain
end
