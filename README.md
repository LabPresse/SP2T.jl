# SP²T.jl

- [SP²T.jl](#sptjl)
  - [Prerequisites](#prerequisites)
    - [Julia](#julia)
    - [GPU](#gpu)
  - [Installation](#installation)
    - [SP²T](#spt)
    - [CUDA.jl](#cudajl)
  - [Quick start](#quick-start)
  - [BNP-Track Framework](#bnp-track-framework)
  - [Citation](#citation)

Single-photon single-particle tracking (SP²T) is the first offline single-particle tracking algorithm to directly track using 1-bit binary frames produced by single-photon detectors such as single-photon avalanche diodes (SPADs).

## Prerequisites

### Julia

SP²T is written in [Julia](https://julialang.org/), so please download and install Julia following its [instructions](https://julialang.org/downloads/). Thanks to Julia's built-in package manager, most packages SP²T relies on do not require manual installation--they will be installed automatically when you install SP²T. The [Installation](#installation) section will list a few exceptions. 

If it interests you, `Project.toml` lists the required Julia packages.

### GPU

We write SP²T so that it can run on CPU or GPU. However, running on GPU is much (>10x) faster. For this reason, you are highly encouraged to try it out.

Although tested mostly on Nvidia GPUs, SP²T itself is not limited to any GPU company as long as the packages it relies on (e.g., [NNlib.jl](https://github.com/FluxML/NNlib.jl)) support the GPU you have.

### Visual Studio Code

This is not a requirement; however, for the best experience, you are encouraged to use an integrated development environment that supports Julia. (Visual Studio Code)[https://code.visualstudio.com/] (VSCode) is the best choice: it is free and the best-supported platform by the Julia community. Please check how to install VSCode and its Julia extension (here)[https://code.visualstudio.com/docs/languages/julia].

## Installation

Once Julia, VSCode, and the Julia extension for VSCode are all installed. It is just one step away from getting SP²T installed. To run the commands in this section, please first open the Julia REPL in VSCode. You can do so by going to "Help>Show All Commands," then typing `Julia: Start REPL` and hitting Enter. 
![Screenshot from 2025-01-09 18-35-53](https://github.com/user-attachments/assets/2a1ab46b-3453-45a7-bbea-d10e09515319)
![Screenshot from 2025-01-09 18-37-43](https://github.com/user-attachments/assets/f64be463-fb84-4ee8-a703-fa0d0cc710a3)
Now Julia REPL should be running, and we are ready to proceed!

### SP²T

SP²T is currently under active development and is still experimental, so it is not yet in Julia's official registry. However, you can try it by typing:

```julia
]add https://github.com/LabPresse/SP2T.jl
```

and hitting Enter in the Julia REPL.

### Other packages

Congrats, you have just installed SP²T! Although nothing else is required in principle, SP²T is designed to be minimal and only to contain the core functions for single-photon tracking. Therefore, it cannot visualize the results and save them to your hard disk. But don't worry; many well-written Julia packages can help us! We need two additional packages to run the example scripts, `Distributions.jl` and `JLD2.jl`. As they are in Julia's official registry, installing them is as easy as `]add Distributions, JLD2`.

#### CUDA.jl

As mentioned in the [Prerequisites](#prerequisites), GPU is highly recommended. Please refer to the [JuliaGPU website](https://juliagpu.org/) for information on what and how to install it based on your hardware.

## Quick start

Once SP²T is installed, you can download the example scripts in `examples/2D`. First, open `simulation.jl` in VSCode; running it will generate some example data. Then, you can open and run either `inference_cpu.jl` or `inference_gpu.jl` for a test calculation using the generated data on CPU or GPU, respectively.

## BNP-Track Framework

The conceptual basis of SP²T is the same as the [BNP-Track](https://www.nature.com/articles/s41592-024-02349-9). The code is modified to cope with single-photon datasets.

## Citation
