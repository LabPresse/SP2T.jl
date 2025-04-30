using SP2T
using Test

println("Running PSF-related tests...")

@test SP2T.gaussianpeak(0.09890858623710756, 0.133) ≈ 0.2486333227348026

@testset let psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixels_size = 0.133,
    )
    @test psf.A ≈ 0.2486333227348026
    @test psf.σ ≈ 0.09890858623710756
end

@testset let (σ₀, z₀) = SP2T.getσ₀z₀(1.45, 1.515, 0.665)
    @test σ₀ ≈ 0.09890858623710756
    @test z₀ ≈ 0.2800714501487841
end