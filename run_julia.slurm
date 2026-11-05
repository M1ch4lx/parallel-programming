#!/bin/bash -l
#SBATCH --job-name=lab4-julia-mpi
#SBATCH --partition=plgrid
#SBATCH --account=UZUPELNIJ_GRANT-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=lab4-%j.out
#SBATCH --error=lab4-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# W razie potrzeby wybierz na Aresie zgodne ze soba wersje Python+MPI+mpi4py.
module load openmpi
module load python

python -c 'from mpi4py import MPI; import mpi4py.futures; print("MPI:", MPI.Get_library_version().splitlines()[0]); print("mpi4py OK")'

# 9 procesow = 1 master + 8 workerow; kazdy proces ma 1 rdzen.
echo '=== PRZEBIEG 1: bez sleep (punkty 1-5) ==='
mpiexec -n "${SLURM_NTASKS}" python -m mpi4py.futures julia.py --mode baseline

# Opoznienie wieksze od najwiekszej sumy liczenia jednego workera w 1. przebiegu.
SLEEP_SECONDS="$(python analysis.py recommend --file baseline.csv)"
echo "Wybrany sleep dla linii 1: ${SLEEP_SECONDS} s"

echo '=== PRZEBIEG 2: sleep (punkt 6) ==='
mpiexec -n "${SLURM_NTASKS}" python -m mpi4py.futures julia.py \
    --mode sleep --sleep-line 1 --sleep-seconds "${SLEEP_SECONDS}"

# Raport dla punktow 2-7; pojedynczy plik z liczbami rzeczywistymi z klastra.
python analysis.py report > raport.txt
cat raport.txt

echo '=== GOTOWE ==='
echo 'Pliki: baseline.pgm, sleep.pgm, baseline.csv, sleep.csv, raport.txt'
