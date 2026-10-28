# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

**SP2T.jl** performs single-particle tracking with frames with discrete and quantized pixel readouts. SP2T was originally developed to work with SPAD array detectors, but then got generalized to be compatible with any frame that can be obtained through adding any integer number of single-photon binary frames.

## Commands

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using SP2T'            # quick load check
```

Test dependencies live in `test/Project.toml` (Julia workspace: `[workspace] projects = ["test"]` in `Project.toml`), not in `[extras]`. CI and `[compat]` target Julia 1.13 only; do not add code paths for older versions.

## Layout

| Path | Contents |
| --- | --- |
| `src/SP2T.jl` | Module, `using` lines, exports, `include` order |
| `src/base.jl` | Abstract types (`Detector`, `PixelDetector`, `PointSpreadFunction`, `RandomVariable`), `LogLikelihoodArray`, array, sampling and proposal helpers |
| `src/distributions.jl` | `DNormal`, the diagonal normal prior on initial positions |
| `src/gaussianpsf.jl` | `CircularGaussian` (2D), `CircularGaussianLorentzian` (3D), pixel integration through `erf` |
| `src/pixeldetector.jl` | Code shared by pixel detectors: construction helpers, expected counts (dark counts + PSF × brightness), likelihood sums over unfiltered pixels |
| `src/spad.jl` | `SPAD`: binomial likelihood over `batchsize` binary frames, readout simulation |
| `src/emccd.jl` | `EMCCD` (Gaussian readout) and `EMCCDGamma` (gamma readout, not exported) |
| `src/msd.jl` | `MeanSquaredDisplacement`, direct sampling from its inverse-gamma posterior |
| `src/emittercount.jl` | `EmitterCount`, direct sampling of the number of emitters up to the weak limit |
| `src/brightness.jl` | `Brightness`, Metropolis–Hastings with a multiplicative random walk |
| `src/tracks.jl` | `TrackChunk`, `MHTrackChunk`, `Tracks`; `simulate!`, `bridge!`; position updates on odd and even frames |
| `src/permutation.jl` | `onshuffle!` (relabels whole on-tracks) and the identity-swap move (relabels tracks from accepted frame boundaries on, so crossing particles can exchange identities) |
| `src/annealing.jl` | `ConstantAnnealing`, `PolynomialAnnealing`, `temperature` |
| `src/chain.jl` | `Sample`, `Chain` (halved when full), `findmap`, `findmle`, `jumpdistances`, `runMCMC`, `runMCMC!`, parametric and nonparametric sweeps |
| `ext/` | `SP2TCUDAExt`: `CuArray` methods of internal functions that would otherwise index scalars |
| `test/runtests.jl` | The test suite (see Testing) |
| `examples/<case>/` | `simulation.jl` writes `frames`, `groundtruth` and `metadata` `.jld2`; `inference_cpu.jl` and `inference_gpu.jl` fit them. `examples/darkcounts.jld2` is shared. Run from the package root; they need JLD2, Distributions and CUDA, which are not package dependencies |
| `backup/`, `extra/`, `data/`, `benchmarks/`, `test/scrach*.jl`, `note.md` | Scratch, gitignored |

## Conventions

- Array shapes: tracks are `(nframes, ndims, nparticles)`, with `ndims` 2 for `CircularGaussian` and 3 (z in column 3) for `CircularGaussianLorentzian`. `readouts` are `(nx, ny, nframes)` `UInt16`; `darkcounts` and `filter` are `(nx, ny)`. `active` is `(nframes, 1, nparticles)` (one entry shared by all coordinates, as in `examples/parametric`) or `(nframes, ndims, nparticles)`. Pixel bounds start at 0, so pixel `(i, j)` covers `[(i-1)·pxsize, i·pxsize] × [(j-1)·pxsize, j·pxsize]`.
- Units are the caller's (the examples use μm and s). `msd` is the per-coordinate variance of a one-frame displacement, 2·D·period; the `Sample` field holding it is named `diffusivity`. `brightness` and `darkcounts` are expected photons per frame (rate × period); `brightness` is divided by `psf.A`, so it is the expected count in a pixel when the emitter sits at its center.
- Everything is parametrized on `T<:AbstractFloat`. Public constructors are `Type{T}(; kwargs...)` and convert their inputs to `T` (`convert`, `elconvert`); methods expect one `T` across all arguments.
- One code path serves `Array` and `CuArray`: allocate with `similar(x, ...)`, avoid scalar indexing, prefer broadcasting and `batched_mul!`. `sum(vec(x))` instead of `sum(x)` is deliberate (faster with CUDA.jl). CUDA-specific methods go in `ext/` as overloads of internal functions; `src/` never loads CUDA.
- Mutating functions end in `!` and return the mutated object. Buffers are preallocated: in `LogLikelihoodArray`, `means[1]` is current and `means[2]` proposed; in `Tracks`, index 1 is current and 2 the proposal.
- `onchunk`, `offchunk` and `proposals` are views into the `Tracks` arrays; call `reassign!` after `nemitters.value` changes. `effvalue = value ./ active` is `Inf` for inactive frames, so they contribute no photons. `active` is a fixed mask; proposals share it (`active[2]` is a separate buffer kept equal to `active[1]`, reserved for inferring the mask later; see F1 in `MEMORY.md`).
- Names: `ᵒ` current, `ᵖ` proposed, `ᵥ` a value (`msdᵥ = msd.value`), `𝑇` temperature, `Δ` proposed minus current. Work in log space. Likelihood terms are divided by `𝑇`, priors are not.
- `RandomVariable` subtypes have `value`, `prior` and `fixed`; `update!` returns at once when `isfixed`. `parametric = true` or a fixed `tracks.nemitters` selects the parametric sweep, which never updates the emitter count. `logprior` of `MeanSquaredDisplacement` and `Brightness` is unnormalized unless `normalization = true`. The track and `DNormal` log-priors are always normalized: their constants scale with the number of emitters, so `log𝒫` would not compare across emitter counts without them.
- Derived properties come from `Base.getproperty` overloads (`chain.msds`, `detector.framecenter`); list new ones in `Base.propertynames` too, as `Chain` does.
- A new detector subtypes `PixelDetector{T}` with fields `period`, `pxsize`, `pxbounds`, `darkcounts`, `filter`, `readouts`, has a keyword constructor built on `check_pixel_dimsmatch` and `_init_pixel_detector_params`, and implements `set_pixel_loglikelihood!`, `set_pixel_Δloglikelihood!` and `simulate_readouts!`.
- The export list in `src/SP2T.jl` is the public API; tests reach internals as `SP2T.name`.
- Style follows `.JuliaFormatter.toml` (4 spaces, 92 columns, trailing commas). Docstrings open with an indented signature line. Don't reformat code you are not changing.

## Working with me

- Make the requested change and nothing more; propose extras instead.
- I edit files between requests. Re-read before editing; treat on-disk changes as intentional unless clearly broken, and then say so.
- Don't commit or push unless asked. Many files are untracked or modified; leave git state alone.
- Scripts, CSVs, MP4s, `scratch.jl`, `results/` and notes at the repository root are my scratch. Don't edit them, run them as tests, or "fix" them.
- Write in American English (normalization, center, parametrize, math), in docs, derivations, code comments and messages.
- Ask before adding dependencies.
- Tests: fixed RNG (`StableRNG` in Julia); derive expected values from theory or a reference implementation, never from current output. A failing test may be a wrong expectation.

## Testing

- `Pkg.test()` runs and passes. `test/runtests.jl` checks the PSF (`SP2T.gaussianpeak`, `CircularGaussian`'s `A` and `σ`, and `SP2T.getσ₀z₀`) against fixed reference values, and the derived properties of `EmitterCount` (`limit` equals the weak limit) and `Chain` (every name in `propertynames` is a `Symbol` that `getproperty` accepts), the track log-prior against `Distributions.logpdf`, `Δlogmotion!` (every frame of the updated parity gets exactly the motion change of its adjacent steps), `Tracks` built with an `active` mask, the `PolynomialAnnealing` schedule (𝑇₀ at the start, 1 from `I` on, decreasing in between), `DNormal.dims`, that detectors leave the caller's `darkcounts` unchanged, that a fixed emitter count is never updated and selects the parametric sweep, and the identity-swap move (frame-by-frame relabeling; two crossing particles with swapped tails are untangled, with their masks), that simulated EMCCD readouts are clipped at 0, and that a parametric step leaves `effvalue` up to date. Detectors, likelihoods, samplers and `runMCMC` have no tests.
- Test dependencies: Test and Distributions. The tests use fixed inputs, no RNG and no data files.
- Tests run on CPU arrays; CI has no GPU, so `ext/` is untested.
- New sampler or fit tests: use a `StableRNG` seed (not yet a test dependency; ask first) or fixed data, and assert against a known ground truth with a tolerance derived from the noise, not tuned to current output.
