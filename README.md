# QTrack / SP²T.jl

- [QTrack / SP²T.jl](#qtrack--sptjl)
  - [Prerequisites](#prerequisites)
    - [Julia](#julia)
    - [GPU](#gpu)
    - [Visual Studio Code](#visual-studio-code)
  - [Installation](#installation)
    - [QTrack / SP²T](#qtrack--spt)
    - [Other packages](#other-packages)
      - [CUDA.jl](#cudajl)
  - [Quick start](#quick-start)
  - [Relation to BNP-Track](#relation-to-bnp-track)
  - [Citation](#citation)

Single-photon single-particle tracking (SP²T) is the first offline single-particle tracking algorithm designed to track particles directly from 1-bit binary frames acquired with single-photon detectors such as single-photon avalanche diodes (SPADs). Because higher-bit frames can be interpreted as sums of underlying 1-bit frames, we extended the same framework to handle measurements with any finite bit depth. This generalized method is referred to as Quantized Track (QTrack).

In other words, SP²T is the 1-bit special case of QTrack, while QTrack refers to the broader framework for direct tracking from quantized measurements with arbitrary finite bit depth.

## Prerequisites

### Julia

QTrack (including its 1-bit special case, SP²T) is written in [Julia](https://julialang.org/), so please download and install Julia (v1.13 or newer) following the official [instructions](https://julialang.org/downloads/). Thanks to Julia’s built-in package manager, most of the packages required by QTrack do not need to be installed manually—they will be installed automatically when you install the package. The [Installation](#installation) section lists a few exceptions.

If helpful, the required Julia packages are listed in `Project.toml`.

### GPU

We wrote QTrack so that it can run on either CPU or GPU. However, GPU execution is typically much faster (often by more than an order of magnitude), so we strongly encourage using a GPU when possible.

Although the code has been tested primarily on Nvidia GPUs, QTrack itself is not restricted to a specific GPU vendor, as long as the Julia packages it depends on (*e.g.*, [NNlib.jl](https://github.com/FluxML/NNlib.jl)) support your hardware.

### Visual Studio Code

This is not required, but for the best experience, we recommend using an integrated development environment with strong Julia support. [Visual Studio Code](https://code.visualstudio.com/) (VS Code) is a particularly good choice: it is free and widely supported by the Julia community. Instructions for installing VS Code and its Julia extension can be found [here](https://code.visualstudio.com/docs/languages/julia).

## Installation

Once Julia, VS Code, and the Julia extension for VS Code are installed, you are just one step away from installing QTrack/SP²T. To run the commands in this section, first open the Julia REPL in VS Code. You can do this by going to **Help > Show All Commands**, then typing `Julia: Start REPL` and pressing **Enter**.

![Screenshot from 2025-01-09 18-35-53](https://github.com/user-attachments/assets/2a1ab46b-3453-45a7-bbea-d10e09515319)
![Screenshot from 2025-01-09 18-37-43](https://github.com/user-attachments/assets/f64be463-fb84-4ee8-a703-fa0d0cc710a3)

Once the Julia REPL is running, you are ready to proceed.

### QTrack / SP²T

QTrack (whose 1-bit special case is SP²T) is currently under active development and is still experimental, so it is not yet included in Julia’s official registry. You can install it directly from GitHub by typing

```julia
]add https://github.com/LabPresse/SP2T.jl
```

in the Julia REPL and pressing **Enter**.

### Other packages

Congratulations — you have now installed QTrack/SP²T. In principle, nothing else is required. However, the package is intentionally kept minimal, containing only the core functions for quantized single-particle tracking. For example, it does not include utilities for visualization or saving results to disk.

To run the example scripts, you will also need two additional packages: Distributions.jl and JLD2.jl. Since both are available in Julia’s official registry, they can be installed easily with

```julia
]add Distributions JLD2
```

#### CUDA.jl

As mentioned in the[Prerequisites](#prerequisites), using a GPU is strongly recommended whenever possible. Please refer to the [JuliaGPU website](https://juliagpu.org/) for instructions on installing and configuring the appropriate GPU stack for your hardware.

## Quick start

Once QTrack/SP²T is installed, you can try the example scripts in `examples/2D`. First, open `simulation.jl` in VS Code and run it to generate example data. Then open and run either `inference_cpu.jl` or `inference_gpu.jl` to perform a test inference on the generated dataset using the CPU or GPU, respectively.

## Relation to BNP-Track

QTrack/SP²T builds on the same general conceptual framework as [BNP-Track](https://www.nature.com/articles/s41592-024-02349-9). In particular, it follows the same Bayesian tracking philosophy, but adapts the implementation to model quantized measurements directly. SP²T corresponds to the 1-bit case, while QTrack extends the framework to data with arbitrary finite bit depth.

## Citation

If this code is useful for your research, please cite the relevant preprint(s):

- **SP²T** — for the 1-bit formulation: https://doi.org/10.1101/2025.01.10.632389  
- **QTrack** — for the generalized finite-bit formulation: https://doi.org/10.1101/2025.05.30.657103
  - The data and code for generating the figures in this paper can be found here: https://github.com/LabPresse/qtrack-paper-figures


