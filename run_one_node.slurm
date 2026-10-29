#!/bin/bash
#SBATCH --job-name=membrane-mpi
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:20:00
#SBATCH --output=membrane_%j.out

# Dostosuj moduły do klastra, np.:
# module load Python OpenMPI
# source .venv/bin/activate

set -euo pipefail

# 1) Mały przypadek: demonstracja + weryfikacja + wizualizacja
mpiexec -n 4 python membrane_mpi.py \
  --n 65 --a 1 --p 1 --tension 1 --tol 1e-8 \
  --verify --plot-prefix results/small --save-npy results/small.npy

# 2) Benchmark dla >=2 rozmiarów problemu i kilku P.
# Wszystkie procesy muszą pozostać na tym samym węźle, bo używamy MPI shared memory.
python benchmark.py \
  --sizes 257,513 \
  --processes 1,2,4,8 \
  --repeats 3 \
  --output results/benchmark.csv

# 3) S(P), E(P), Karp-Flatt f(P)
python plot_benchmarks.py \
  --input results/benchmark.csv \
  --output-dir results/plots
