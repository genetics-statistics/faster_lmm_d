# Faster_lmm_d

A faster lmm for GWAS. GPU support coming soon.

NOTICE: this software is under active development. YMMV.

# Install

## Requirements

`faster_lmm_d` is written in the fast D language and requires a D
compiler. At the moment we also use openblas, lapacke and gsl
libraries.

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
guix package -i ldc dub openblas gsl lapack
```

## Get the source

Get the source-code

```sh
git clone https://github.com/prasunanand/faster_lmm_d
cd faster_lmm_d
```

Fetch dependencies and compile

```sh
dub --compiler=ldc2
```

or in the case of GNU Guix (because dub does not honour the
LIBRARY_PATH):

```sh
env LD_LIBRARY_PATH=$LIBRARY_PATH dub --compiler=ldc2
```

Usage example

```sh
./faster_lmm_d --control=data/genenetwork/BXD.json --pheno=data/genenetwork/104617_at.json --geno=data/genenetwork/BXD.csv --cmd=rqtl
```

To run tests

```sh
./run_tests.sh
```

## LICENSE

This software is distributed under the [GPL3 license](https://www.gnu.org/copyleft/gpl.html).

Copyright © 2016, 2017, Prasun Anand and Pjotr Prins
