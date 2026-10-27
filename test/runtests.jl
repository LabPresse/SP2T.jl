using SP2T
using Test

println("Running PSF-related tests...")

@test SP2T.gaussianpeak(0.09890858623710756, 0.133) ≈ 0.2486333227348026

@testset let psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixel_size = 0.133,
    )
    @test psf.A ≈ 0.2486333227348026
    @test psf.σ ≈ 0.09890858623710756
end

@testset let (σ₀, z₀) = SP2T.getσ₀z₀(1.45, 1.515, 0.665)
    @test σ₀ ≈ 0.09890858623710756
    @test z₀ ≈ 0.2800714501487841
end

println("Running property-related tests...")

@testset let count = SP2T.EmitterCount{Float64}(1, 10, -10)
    @test count.limit == 10
end

@testset let chain = Chain(
        [Sample(zeros(3, 2, 1), 1.0, 1.0, 0, 1.0, NaN, NaN)],
        10,
        ConstantAnnealing{Float64}(1),
    )
    @test all(s -> s isa Symbol, propertynames(chain))
    @test fieldnames(Chain) ⊆ propertynames(chain)
    for s in propertynames(chain)
        @test (getproperty(chain, s); true)
    end
end