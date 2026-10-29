mpiexec -n 1 -- python membrane_mpi.py \
    --n 257 \
    --a 1.0 \
    --p 1.0 \
    --tension 1.0 \
    --tol 1e-8 \
    --max-iters 200000 \
    --json