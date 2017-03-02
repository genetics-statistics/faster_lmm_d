# Faster_lmm_d

A faster lmm for GWAS. GPU support coming soon.

# Install

## Debian/Ubuntu

Install BLAS, LAPACK, GSL, OpenCL.

```sh
sudo apt-get install libblas-common liblapacke libgsl2 ocl-icd-libopencl1
```

Install LDC

```
sudo apt-get install ldc2
```

## GNU Guix

```sh
guix package -i ldc dub lapack
```

## Get the source

Get the source-code

```sh
git clone https://github.com/prasunanand/faster_lmm_d
cd faster_lmm_d
```

Compiling
```sh
dub --compiler=ldc2
```

To use
```sh
./faster_lmm_d --control=data/genenetwork/BXD.json --pheno=data/genenetwork/104617_at.json --geno=data/genenetwork/BXD.csv --cmd=rqtl
```

To run test8000
```sh
bash test.sh
```
