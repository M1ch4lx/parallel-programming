#!/bin/bash -l

#SBATCH -A plgar2026-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -J lab4-julia-mpi
#SBATCH -o lab4-%j.out
#SBATCH -e lab4-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

# Srodowisko z dzialajacego benchmarku
module load scipy-bundle/2021.10-intel-2021b

# Kontrola dostepnosci MPI i mpi4py
if ! python -c 'from mpi4py import MPI; import mpi4py.futures' >/dev/null 2>&1; then
    echo "BLAD: Brak mpi4py lub niezgodny mpi4py." >&2
    echo "Sprawdz: module spider mpi4py" >&2
    echo "Zaladuj modul zgodny z Intel 2021b." >&2
    exit 1
fi

if ! command -v mpiexec >/dev/null 2>&1; then
    echo "BLAD: Brak mpiexec." >&2
    exit 1
fi

for file in julia.py analysis.py; do
    if [[ ! -f "$file" ]]; then
        echo "BLAD: Brak $file w katalogu $PWD" >&2
        exit 1
    fi
done

python -c 'from mpi4py import MPI; print("MPI:", MPI.Get_library_version().splitlines()[0]); print("mpi4py OK")'

echo "=== PRZEBIEG 1: bez sleep ==="

mpiexec -n "${SLURM_NTASKS}" \
    python -m mpi4py.futures julia.py --mode baseline

# Dobor sleep na podstawie pierwszego przebiegu
SLEEP_SECONDS="$(python analysis.py recommend --file baseline.csv)"

echo "Wybrany sleep: ${SLEEP_SECONDS} s"

echo "=== PRZEBIEG 2: z sleep ==="

mpiexec -n "${SLURM_NTASKS}" \
    python -m mpi4py.futures julia.py \
    --mode sleep \
    --sleep-line 1 \
    --sleep-seconds "${SLEEP_SECONDS}"

echo "=== GENEROWANIE RAPORTU ==="

python analysis.py report > raport.txt

cat raport.txt

echo "=== GOTOWE ==="
echo "Pliki: baseline.csv, sleep.csv, baseline.pgm, sleep.pgm, raport.txt"
