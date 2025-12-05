
# Overview

This code performs a sparse matrix sparse matrix multiplication (SpMSpM).

# Instructions

To compile:

```
make
```

To run:

```
./spmspm [flags]

```

Optional flags:

```
  -f <matrixFile>   name of the file with the matrix to multiply (the matrix must be 
                    square and will be multiplied with itself)

  -0                run GPU version 0
  -1                run GPU version 1
  -2                run GPU version 2
  -3                run GPU version 3
                    NOTE: It is okay to specify multiple different GPU versions in the
                          same run. By default, only the CPU version is run.

  -v                perform exact verification of the GPU run across the CPU run

```

