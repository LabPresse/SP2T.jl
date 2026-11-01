# MEMORY.md

Open issues from the code review of 2026-10-27. Each was confirmed on CPU unless it is under "Needs a GPU" (the Mac has no CUDA, CI has no GPU). Delete an entry once it is settled.

## Performance (CPU)

2D example (50×50 pixels, 2550 frames, 10 candidates, Float32), after P1: one nonparametric iteration takes 459 ms on CPU and 166 ms on Metal (M1 Pro). Starting from the true tracks (2 on), the CPU spends 270 ms in the emitter-count update, 147 ms in the position update and 61 ms in the brightness update; one expected-count evaluation (`set_poisson_mean!`) takes 17 ms (copying `darkcounts` 4 ms, PSF integration 10 ms, `batched_mul!` 1 ms). P2 (preallocating the PSF buffers) was measured and dropped: allocation takes 0.01–0.04 ms per call.

### P3. Recomputed current expected counts

`update_onchunk!` and `update!(::Brightness, ...)` recompute the expected counts of the current state, which the previous step already had. Caching them would save one `set_poisson_mean!` (17 ms on CPU) in each, about 7% of an iteration, at the cost of keeping the cache valid after every accepted move, emitter-count change and brightness change.

### P4. Emitter-count likelihood over all pixels

The emitter-count update evaluates the full log-likelihood once per candidate (`max_ntracks` times per iteration), over every pixel and frame, though adding one candidate only changes pixels near its track. Restricting each evaluation to a window around the candidate (a few PSF widths) would cut most of the 270 ms, but needs per-candidate windows that also work on GPUs.

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
