# SP²T.jl

- [SP²T.jl](#sptjl)
  - [Prerequisites](#prerequisites)
    - [Julia](#julia)
    - [GPU](#gpu)
  - [Installation](#installation)
    - [SP²T](#spt)
    - [CUDA.jl](#cudajl)
  - [BNP-Track Framework](#bnp-track-framework)
  - [Citation](#citation)

Single-photon single-particle tracking (SP²T) is the first offline single-particle tracking algorithm to directly track using 1-bit binary frames produced by single-photon detectors such as single-photon avalanche diodes (SPADs).

## Prerequisites

### Julia

SP²T is written in [Julia](https://julialang.org/), so please install it following its instructions. Thanks to Julia's built-in package manager, users usually are not required to install anything themselves. Please have a look at `Project.toml` if you want to check what Julia packages are used by SP²T.

### GPU

We write SP²T so that it can run on CPU or GPU. However, running on GPU is much (>10x) faster. For this reason, you are highly encouraged to try it out.

Although tested mostly on Nvidia GPUs, SP²T itself is not limited to any GPU company as long as the packages it relies on (e.g., [NNlib.jl](https://github.com/FluxML/NNlib.jl)) support the GPU you have.

## Installation

### SP²T

SP²T is currently under active development and is still experimental, so it is not yet in Julia's official registry. However, you can try it by typing

```julia
]add https://github.com/LabPresse/SP2T.jl
```

in the Julia REPL.

### CUDA.jl

As mentioned in the [Prerequisites](#prerequisites), using GPU is highly recommended. Please refer to the [JuliaGPU website](https://juliagpu.org/) for what and how to install based on your hardware.

## BNP-Track Framework

The conceptual basis of SP²T is the same as the [BNP-Track](https://www.nature.com/articles/s41592-024-02349-9). The code is modified to cope with single-photon datasets.

## Citation
