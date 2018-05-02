# Faster_lmm_d

[![Build Status](https://travis-ci.org/prasunanand/faster_lmm_d.svg?branch=master)](https://travis-ci.org/prasunanand/faster_lmm_d)

A faster lmm for GWAS. It has multi-core and GPU support.

NOTICE: this software is under active development. YMMV.

# Introduction

Faster_lmm_d is a lightweight linear mixed-model solver for use in
genome-wide association studies (GWAS). The original is similar to
FaST-LMM (an algorithm by Lippert et al.) and that code base can be
found [here](https://github.com/nickFurlotte/pylmm). Prof. Karl Broman
wrote a comparison with his
[R/lmmlite](http://kbroman.org/lmmlite/assets/lmmlite.html). Faster_lmm_d
and pylmm are part of the
[Genenetwork2](https://github.com/genenetwork) project. faster_lmm_d
can parse data in
[R/qtl2 format](http://kbroman.org/qtl2/assets/vignettes/input_files.html)
as input.

# GPU Support

Faster_lmm_d has two GPU backends:

CUDA backend which helps it directly interact with CUBLAS libraries and runs
only on Nvidia Hardware. For CUDA backend, Faster_LMM_D uses [cuda_d](https://github.com/prasunanand/cuda_d)
(The D bindings I wrote for CUDA libraries).

ArrayFire backend which helps it run on all major GPU vendors(Nvidia, Intel, AMD)
by calling CUDA, CuBLAS, OpenCL, clBLAS libraries using the ArrayFire library.
For ArrayFire backend, Faster_LMM_D uses [arrayfire-d](https://github.com/arrayfire/arrayfire-d)
(The D bindings I wrote for ArrayFire library).

# Install

## Requirements

`faster_lmm_d` is written in the fast D language and requires a D
compiler. At the moment we also use openblas (>0.2.19), lapacke, gsl
and a bunch of D libraries.

### On Debian/Ubuntu

Install

```sh
sudo apt-get install libopenblas liblapacke libgsl2
```

Install LDC

```
sudo apt-get install ldc2
```

### On GNU Guix

```sh
guix package -i ldc dub openblas gsl lapack ld-wrapper gcc glibc
```

## Get the source

Get the source-code

```sh
git clone https://github.com/prasunanand/faster_lmm_d
cd faster_lmm_d
```

Fetch dependencies and compile

CPU Backend:
```sh
make
```
CUDA Backend:
```sh
make CUDA=1
```
ARRAYFIRE Backend:
```sh
make ARRAYFIRE=1
```

or in the case of GNU Guix (because dub does not honour the
LIBRARY_PATH):

```sh
export LIBRARY_PATH=~/.guix-profile/lib
env LD_LIBRARY_PATH=$LIBRARY_PATH dub --compiler=ldc2
```

Usage example

```sh
./faster_lmm_d --control=data/genenetwork/BXD.json --pheno=data/genenetwork/104617_at.json --geno=data/genenetwork/BXD.csv --cmd=rqtl
```

## Testing

To run tests

```sh
time ./run_tests.sh
```

If you get an error (on GNU Guix)

```sh
./build/faster_lmm_d: error while loading shared libraries: libgsl.so.19: cannot open shared object file: No such file or directory
```

try

```sh
time env LD_LIBRARY_PATH=$LIBRARY_PATH ./run_tests.sh
```

## Performance Profiling

Install google-perftools and graphviz

```sh
sudo apt-get install google-perftools libgoogle-perftools-dev graphviz
```

Install go and then install google-pprof.

```sh
go get github.com/google/pprof
```

To profile uncomment out the code import `gperftools_d.profiler;`, `ProfilerStart()` and `ProfilerStop()` in
the `main` function in `source/faster_lmm_d/app.d`.

```sh
make run-gperf
```

## Useful links:

Poster on [Faster Linear Mixed Models (LMM) for online GWAS omics analysis​](https://github.com/prasunanand/resume/blob/master/CTC_2017_Poster_Faster_LMM_D.pdf) at Complex Trait Community Conference 2017, Memphis, Tennessee, USA.

## LICENSE

This software is distributed under the [GPL3 license](https://www.gnu.org/copyleft/gpl.html).

Copyright © 2016 - 2018, Prasun Anand and Pjotr Prins
