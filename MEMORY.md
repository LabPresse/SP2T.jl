# MEMORY.md

Open issues from the code review of 2026-10-27. Each was confirmed on CPU unless it is under "Needs a GPU" (the Mac has no CUDA, CI has no GPU). Delete an entry once it is settled.

## Legacy code and comments

### L1. Unused code

- `unionalltypeof` (`src/base.jl`)
- 3-argument `_randn!(x, σ, σ₀)` (`src/base.jl`)
- `mrw_propose` for `BetaPrime`/`Gamma`, marked "Needs test"; `Brightness.proposal` is always a `Beta`
- `ftypof`, scalar `anneal(ll, 𝑇)` (`src/annealing.jl`)
- `_integrate` (`src/gaussianpsf.jl`); worth keeping for a PSF test
- `bridge!`: exported, undocumented, used nowhere

### L2. Comments

- Commented-out `StreamDetector` (`src/base.jl`)
- Commented-out `# get_Δloglikelihood!` (`src/brightness.jl`)
- "used mostly for prototyping" in the `RandomVariable` docstring
- `#* This code is optizmized for GPU.` (`src/SP2T.jl`)

### L3. Wrong or outdated docstrings and messages

- `Brightness`: says the proposal default is `(10, 1)` (code: `(1, 10)`), lists `fixed` twice, says `P<:Gamma{T}` (unconstrained in the code)
- `_init_pixel_detector_params`: documents a `period` argument that does not exist
- `MeanSquaredDisplacement` constructor: stray `'` instead of a backtick
- Typos: "dose not", "categoriacal", "implememnt", "distrtibution", "mltiplicative", "Gaussain", "sames"

### L4. Other

- `@showprogress 1 "Computing..."` is deprecated ProgressMeter syntax and prints a warning; use `@showprogress dt = 1 desc = "Computing..."`.
- `AbstractEMCCD` has one subtype; `EMCCDGamma` subtypes `PixelDetector` directly.
- `findmap` and `findmle` take `burn_in::Real`; should be `Integer`.
- `ext/SP2TCUDAExt.jl` loads `Random` without using it.

## Tests to add

All on CPU, with expected values from theory or Distributions.jl:

- T1. Detector likelihoods against Distributions.jl up to terms that do not depend on the mean: SPAD vs `Binomial(batchsize, 1 − exp(−λ))`, EMCCD vs `Normal`, EMCCDGamma vs `Gamma`. Each Δ version equals the difference of two full versions.
- T2. PSF: the integrated PSF sums to ≈ 1 over a wide window; the pixel under an emitter at its center gets `darkcount + brightness`.
- T3. `framesum!`, `diff²!` and `staggered_diff²!` against direct formulas.
- T6. `Chain`: thinning keeps at most `sizelimit` samples at equal spacing; `findmap` and `findmle` respect `burn_in`; `jumpdistances` on known tracks.
- T7. Sampler tests need an `rng` argument: the code uses the global RNG, so a `StableRNG` cannot be passed in.

## Performance (CPU)

Measured with 30×30 pixels, 1000 frames, 8 candidates: one nonparametric iteration takes 262 ms, of which the emitter-count update takes 183 ms, the position update 67 ms and the brightness update 41 ms. One `set_poisson_mean!` takes 9 ms, one SPAD `set_pixel_loglikelihood!` 22 ms.

### P1. Skip `logexpm1` for zero SPAD readouts

`readouts * logexpm1(mean)` evaluates `logexpm1` for every pixel, though most SPAD readouts are 0. A real branch, `iszero(r) ? zero(m) : r * logexpm1(m)`, gives identical results and is 9.4× faster (12.5 ms → 1.3 ms, 97% zeros). `ifelse` evaluates both arms and is not faster. Applies to `set_pixel_loglikelihood!` and `set_pixel_Δloglikelihood!`. GPU effect: see G4.

### P2. Per-call allocations

`set_poisson_mean!`, `set_loglikelihood!` (emitter count) and `getincident` allocate `psfcomponents` and `relativepos` on every call (≈ 1.9 MiB per call above; several calls per iteration). Preallocate once, e.g. in `LogLikelihoodArray`.

### P3. Recomputed current expected counts

`update_onchunk!` and `update!(::Brightness, ...)` recompute the expected counts of the current state, which the previous step already had. Caching them saves one PSF evaluation per update.

## Future

### F1. Infer the `active` mask

`active` is currently a fixed, known mask. Ideally it becomes a random variable (blinking on/off states). `Tracks.active[2]`, the proposals' mask, is kept as a separate buffer equal to `active[1]` so a mask sampler can later propose masks there without changing the data layout.

### F2. One switch for a fixed emitter count

`runMCMC` and `runMCMC!` run the parametric sweep when `parametric = true` or `tracks.nemitters.fixed` is set. Consider making `fixed` the only switch: drop `parametric`, add a `Tracks` keyword to fix the count, so every random variable says whether it is sampled and the sweep follows (as `MeanSquaredDisplacement` and `Brightness` already do, and as F1 would for the mask). Breaking change: every example passes `parametric`, several positionally; do it when the examples are next updated.

## Needs a GPU

### G1. `ext/` methods may never be called

`SP2TCUDAExt` defines `_permute!(x::CuArray{T,3}, ...)` and `addΔlogprior₁!(ln𝓇::CuVector{T}, x::CuArray{T,3}, y::CuArray{T,3}, ...)`. The arguments passed in are `@views` slices such as `value[:, :, 1:nemitters]`. These match only if CUDA.jl returns a `CuArray` (not a `SubArray`) for a contiguous view. Otherwise the generic methods run: `logacceptance[1] += ...` indexes a scalar (an error outside the REPL), and `src[:, :, perm]` gathers with a CPU index vector.

Check: on a GPU run, `typeof(tracks.onchunk.value)` and `typeof(tracks.proposals.value)` are `CuArray`. Run a short `runMCMC` with `CUDA.allowscalar(false)`.

### G2. Nested array wrappers in the PSF

`integratepsf!` passes `view(tracksᵥ, :, 1:1, :)` to `_erf!`, which broadcasts over `PermutedDimsArray` of it; `lateral_std` does the same for z. That is a `PermutedDimsArray{SubArray{CuArray}}`. Check that this broadcast compiles to one GPU kernel (not a scalar fallback), with `CUDA.allowscalar(false)`, and time it against materializing `permutedims` first.

### G4. P1 on GPU

The zero-readout branch diverges within a warp, so it may gain less or nothing. Time both versions on a realistic data set before changing the shared code path.

### G5. Host synchronizations per iteration

Each of these forces a GPU → CPU copy and a synchronization:

- `get_loglikelihood!` returns `sum(llarray.frame)`; the emitter-count update calls it once per candidate (`max_ntracks` times per iteration).
- `addΔlogprior₁!` (`CUDA.@allowscalar`).
- `countacceptance!` (`count(>(0), t.accepted)`).
- `onshuffle!` → `randperm` on CPU, then one `CUBLAS.swap!` per emitter.
- The identity-swap move (`update!(::TrackChunk, ::MHTrackChunk, msd)`): copies `accepted` to the CPU, builds the relabeling indices there and uploads them (`nframes × ndims × nemitters` integers), then gathers with `scratch[i]`. Check that the gather runs as one kernel (it needs `scratch` to be a `CuArray`, see G1) and what the transfers cost; the CPU cost is 0.1–1.2 ms per call at 2550 frames and 10 particles.

Profile one iteration with `CUDA.@profile` (or Nsight Systems) to see whether these, rather than kernels, dominate.

### G6. P2 on GPU

On GPU the per-call buffers come from the memory pool. Check with `CUDA.@time` whether allocation counts or pool pressure matter.

### G7. `sum(vec(x))` instead of `sum(x)`

A comment in `src/msd.jl` says summing `vec` is faster with CUDA.jl. Recheck with the current CUDA.jl; if the difference is gone, use `sum(x)` and drop the convention from AGENTS.md.
