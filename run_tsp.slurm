#!/bin/bash -l
#SBATCH --job-name=tsp-bb-mpi
#SBATCH -A plgar2026-cpu
#SBATCH -p plgrid
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=tsp-bb-%j.out
#SBATCH --error=tsp-bb-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Tak jak w przeslanym run_julia.slurm; wersje modulow moga sie roznic
# w zaleznosci od konfiguracji srodowiska na Aresie.
module load openmpi
module load python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

python -c 'from mpi4py import MPI; from mpi4py.futures import MPIPoolExecutor; import matplotlib; print("MPI:", MPI.Get_library_version().splitlines()[0]); print("matplotlib:", matplotlib.__version__)'

# 9 procesow = 1 master i do 8 workerow.
# Program pomiarowy uruchamia mpiexec osobno dla kazdej pary (d,p).
# Eksperyment 1: d=3, p=1,2,4,8.
# Eksperyment 2: p=8, d=2,3,4 (plus osobne bazowe p=1 dla kazdego d).
# Caly czas jest mierzony na zewnatrz mpiexec, wiec generowanie prefiksow
# na pewno wchodzi do pomiarow, podobnie jak start/stop MPI.
python benchmark_tsp.py \
    --n 13 --seed 2026 --radius 100 --bound minedge \
    --depth-fixed 3 --workers 1,2,4,8 \
    --workers-fixed 8 --depths 2,3,4 \
    --repeats 3 --outdir "wyniki_tsp_${SLURM_JOB_ID}"

echo '=== GOTOWE ==='
echo "Zobacz: wyniki_tsp_${SLURM_JOB_ID}/raport.md i 6 wykresow PNG."
