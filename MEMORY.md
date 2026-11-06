# MEMORY.md

Open issues from the code review of 2026-10-27. Each was confirmed on CPU unless it is under "Needs a GPU" (the Mac has no CUDA, CI has no GPU). Delete an entry once it is settled.

## Bugs

### B11. NaN in dark frames with the 3D PSF

A dark frame (`active` 0) has `effvalue = Inf` in every coordinate, z included. `lateral_std` then gives σ = Inf, and `_erf!` computes (bounds − Inf)/(√2·Inf) = NaN, so with `CircularGaussianLorentzian` every dark frame's likelihood is NaN (the 2D PSF gives 0, as intended). NaN log-acceptances are never accepted and NaN emitter-count posteriors break `randc`. Any 3D run with an `active` mask is affected. Possible fix: give dark frames a finite σ (e.g. σ₀ where z is infinite), so the relative positions are −Inf and the PSF fractions 0. The SPAD equivalence test in `test/runtests.jl` keeps its dark frame out of the 3D case until this is fixed.

## Performance (CPU)

2D example (50×50 pixels, 2550 frames, 10 candidates, Float32), starting from the true tracks (2 on): one nonparametric iteration took 459 ms on CPU and 166 ms on Metal (M1 Pro) after P1, and takes 58 ms on CPU and 33 ms on Metal with the PSF cutoff (`pixelfraction`) and the sparse SPAD workspace (`SPADLogLikelihood`). The dense workspace with the cutoff takes 303 ms on CPU. P2 (preallocating the PSF buffers) was measured and dropped: allocation takes 0.01–0.04 ms per call.

### P3. Recomputed current expected counts

`update_onchunk!` and `update!(::Brightness, ...)` recompute the expected counts of the current state, which the previous step already had. Caching them would save one `set_poisson_mean!` in each, at the cost of keeping the cache valid after every accepted move, emitter-count change and brightness change. Before the sparse workspace this was about 7% of an iteration; re-measure on the sparse path (where one evaluation is mostly PSF integration) before deciding.

### P4. Emitter-count likelihood over all pixels (EMCCD)

With the dense workspace (EMCCD, EMCCDGamma), the emitter-count update evaluates the full log-likelihood once per candidate (`max_ntracks` times per iteration), over every pixel and frame, though adding one candidate only changes pixels near its track. Restricting each evaluation to a window around the candidate (a few PSF widths) would cut most of it, but needs per-candidate windows that also work on GPUs. SPAD no longer needs this: its sparse workspace touches only nonzero readouts.

## Sampling and summaries (planned, in this order)

Motivation (2026-11-02): from its own starting values (tracks at zero, brightness 0.1, MSD 2e-5), the 2D example settles within ~120 iterations into 10 dim emitters (brightness ≈ 0.015, MSD ≈ 9e-6) 0.4–0.6 μm from the particles, and stays there for 3000 iterations. Started at the ground truth it stays there (2 on, brightness ≈ 0.098, MSD ≈ 2e-4). The truth's log-likelihood is higher by ~5,300. So both are stable modes and the chain cannot travel between them. All moves below must leave the posterior unchanged (`docs/logposterior.qmd`); test each by sampling the prior with the likelihood switched off (𝑇 = Inf).

### S2. Interweaving MSD move (2a)

Implemented 2026-11-03 (`update!(rng, ::MeanSquaredDisplacement, ::Tracks, ...)` in `src/tracks.jl`, `docs/msdrescale.qmd`): after the Gibbs draw, propose msd′ = msd·exp(τz) (`proposalscale` τ, default 0.05) and stretch the on tracks around their first positions by √(msd′/msd); accepted with the likelihood ratio, the MSD prior ratio and msd′/msd. In the ablation (S1) it took convergence from 1/3 to 3/3 chains for +1–5 ms per iteration, as did the joint count–brightness move (2b); neither frees the 2D example from its own start. Over five data sets (23 chains each, 2026-11-05), converged chains: none 20, 2a 22, 2b 20, 2a+2b 23; converging chains took similar CPU time (2a: fewer iterations, ~25% more CPU per iteration), and 2b's stuck chains split the brightness over 4–6 emitters. So since 2026-11-06 both are off by default (`runMCMC` keywords `msdrescale`, `countbrightness`). Started at the truth (2026-11-06; 5 data sets × 2 seeds × 19,000 kept iterations), neither raised the effective sample size per iteration of the MSD (median 25–29 for every variant), brightness (~2,800), positions (50–70) or log-likelihood, so per CPU second the plain sweep was best (2a: 20–40% fewer effective samples per second, equal for the log-likelihood; its acceptance 35%). The MSD's ESS (~1 per 700 iterations) follows the positions' slow mixing, which is the bottleneck after convergence. Open: the stretch about the first frame moves late frames a lot, so long tracks may reject it; a stretch about each track's mean, or per segment, would move them less.

### S3. Coordinate-invariant posterior summaries (postprocessing)

From a chain: the emitter count's posterior probabilities; medians and credible intervals of the MSD and brightness; per-frame track summaries (mean positions, localization maps), which first need labels resolved across samples. `findmap` stays as a coordinate-dependent point estimate (its docstring says so).

## Future

### F1. Infer the `active` mask

`active` is currently a fixed, known mask. Ideally it becomes a random variable (blinking on/off states). `Tracks.active[2]`, the proposals' mask, is kept as a separate buffer equal to `active[1]` so a mask sampler can later propose masks there without changing the data layout.

### F2. One switch for a fixed emitter count

`runMCMC` and `runMCMC!` run the parametric sweep when `parametric = true` or `tracks.nemitters.fixed` is set. Consider making `fixed` the only switch: drop `parametric`, add a `Tracks` keyword to fix the count, so every random variable says whether it is sampled and the sweep follows (as `MeanSquaredDisplacement` and `Brightness` already do, and as F1 would for the mask). Breaking change: every example passes `parametric`, several positionally; do it when the examples are next updated.

### F3. Reactant.jl (low priority)

Compiling the sweeps with Reactant.jl (XLA) could fuse kernels on CPU and GPU, but needs fixed array shapes: always compute all `max_ntracks` candidates and switch them with a mask (the `active`/`effvalue = Inf` design points that way), and move host-side decisions (`randperm`, scalar accept/reject, Distributions.jl draws) into traced code. Reactant's Metal support is experimental (through Apple's JAX Metal plugin). Revisit only as part of such a fixed-shape redesign.

## Needs a GPU

Timing single small GPU operations with host timers is unreliable on Metal (results varied by 100× between runs of the same operation, likely GPU power states); time whole steps, or use a GPU profiler (Xcode Instruments, Metal capture). On the Mac, Metal (an Apple GPU) can stand in for CUDA to check items that only depend on GPUArrays behavior (G1, G2, G8); a pass on Metal is evidence, not proof, for CUDA.

### G1. `ext/` methods may never be called

`SP2TCUDAExt` defines `_permute!(x::CuArray{T,3}, ...)` and `addΔlogprior₁!(ln𝓇::CuVector{T}, x::CuArray{T,3}, y::CuArray{T,3}, ...)`. The arguments passed in are `@views` slices such as `value[:, :, 1:nemitters]`. These match only if CUDA.jl returns a `CuArray` (not a `SubArray`) for a contiguous view. Otherwise the generic methods run: `logacceptance[1] += ...` indexes a scalar (an error outside the REPL), and `src[:, :, perm]` gathers with a CPU index vector.

Check: on a GPU run, `typeof(tracks.onchunk.value)` and `typeof(tracks.proposals.value)` are `CuArray`. Run a short `runMCMC` with `CUDA.allowscalar(false)`.

### G2. Nested array wrappers in the PSF

`integratepsf!` passes `view(tracksᵥ, :, 1:1, :)` to `_erf!`, which broadcasts over `PermutedDimsArray` of it; `lateral_std` does the same for z. That is a `PermutedDimsArray{SubArray{CuArray}}`. Check that this broadcast compiles to one GPU kernel (not a scalar fallback), with `CUDA.allowscalar(false)`, and time it against materializing `permutedims` first.

### G5. Host synchronizations per iteration

Each of these forces a GPU → CPU copy and a synchronization:

- `get_loglikelihood!` returns `sum(llarray.frame)`; the emitter-count update calls it once per candidate (`max_ntracks` times per iteration).
- `addΔlogprior₁!` (`CUDA.@allowscalar`).
- `countacceptance!` (`count(>(0), t.accepted)`).
- `onshuffle!` → `randperm` on CPU, then one `CUBLAS.swap!` per emitter.
- The identity-swap move (`update!(::TrackChunk, ::MHTrackChunk, msd)`): copies `accepted` to the CPU, builds the relabeling indices there and uploads them (`nframes × ndims × nemitters` integers), then gathers with `scratch[i]`. Check that the gather runs as one kernel (it needs `scratch` to be a `CuArray`, see G1) and what the transfers cost; the CPU cost is 0.1–1.2 ms per call at 2550 frames and 10 particles.

Profile one iteration with `CUDA.@profile` (or Nsight Systems) to see whether these, rather than kernels, dominate.

### G8. Random numbers on GPU

Array draws go through `devicerng(rng, x)`, which `SP2TCUDAExt` maps to `CUDA.default_rng()` for a `CuArray` and, through `parent`, for a view of one. Check that `randn!`, `rand!` and `randexp!` with `CUDA.default_rng()` work on the views the samplers pass (e.g. `x[2:end, :, :]` in `simulate!`), and that seeding `CUDA.default_rng()` plus the host `rng` makes a GPU run reproducible. The SPAD (`batchsize > 1`) and `EMCCDGamma` simulations broadcast `rand.(Ref(rng), ...)` with Distributions.jl, which runs on CPU only, as before.

### G7. `sum(vec(x))` instead of `sum(x)`

A comment in `src/msd.jl` says summing `vec` is faster with CUDA.jl. Recheck with the current CUDA.jl; if the difference is gone, use `sum(x)` and drop the convention from AGENTS.md.

### G9. Sparse SPAD workspace on CUDA

`SPADLogLikelihood` gathers the PSF components at the clicks with integer index arrays (`px[xindex]`), sums per frame with `sum!` over a reshaped vector, and builds its index arrays on the CPU, copying them to the device once per `runMCMC!` call. It passes on Metal (`test/metal.jl`); check on CUDA with `CUDA.allowscalar(false)` that the gathers run as kernels, and time an iteration against the dense workspace (`LogLikelihoodArray`) there.
