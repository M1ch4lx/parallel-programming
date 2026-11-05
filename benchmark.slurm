#!/bin/bash -l

#SBATCH -A plgar2026-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 06:00:00
#SBATCH -J membrane
#SBATCH -o benchmark-%j.out
#SBATCH -e benchmark-%j.err

module load scipy-bundle/2021.10-intel-2021b

export SLURM_OVERLAP=1

cd "$SLURM_SUBMIT_DIR"

mkdir -p results



python -u benchmark.py \
    --sizes 513,1025,1537 \
    --processes 1,2,4,8,16 \
    --repeats 1 \
    --tol 1e-30 \
    --max-iters 1500 \
    --progress-every 300 \
    --output results/benchmark.csv

python plot_benchmarks.py --input results/benchmark.csv --output-dir results/plots