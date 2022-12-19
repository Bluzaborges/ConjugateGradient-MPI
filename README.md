## Parallelized conjugate gradient method using MPI

A simple implementation of conjugate gradient method with the objective of parallelize using the MPI library in C.

The solution must be compiled with the following command:

```
mpicc -o gradient gradient.c 
```

And executed with:

```
mpirun -np <number of process> gradient <order of matrix>
```