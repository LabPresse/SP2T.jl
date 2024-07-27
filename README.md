# SP²T

- [SP²T](#spt)
  - [Installation](#installation)
  - [BNP-Track Framework](#bnp-track-framework)
  - [Citation](#citation)

Single-photon single-particle tracking (SP²T) is the first offline single-particle tracking algorithm to directly track using 1-bit binary frames produced by single-photon detectors such as single-photon avalanche diodes (SPADs).

## Installation

SP²T is currently under active development and is still experimental, so it is not yet in Julia's official registry. However, you can try it by typing

```julia
]add add https://github.com/LabPresse/SP2T.jl
```

in your Julia REPL.

## BNP-Track Framework

The conceptual basis of SP²T is the same as the [BNP-Track](https://www.nature.com/articles/s41592-024-02349-9). The code is modified to cope with single-photon datasets.

## Citation