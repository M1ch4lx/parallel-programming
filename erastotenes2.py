#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import math
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Sequential sieve
def sieve_base(limit):
    if limit < 2:
        return []

    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0:2] = False

    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p:limit + 1:p] = False

    return np.flatnonzero(is_prime).tolist()


if len(sys.argv) != 2:
    if rank == 0:
        print("Usage:")
        print("mpiexec -n NUMBER_OF_PROCESSES python3 sieve_mpi.py n")

    sys.exit(1)


n = int(sys.argv[1])

if n < 2:
    if rank == 0:
        print("There are no prime numbers in the given range")
    sys.exit(0)


sqrt_n = math.isqrt(n)

divisors = sieve_base(sqrt_n)


c_start = sqrt_n + 1
c_end = n

total_size = c_end - c_start + 1


chunk_size = total_size // size
remainder = total_size % size


if rank < remainder:

    local_size = chunk_size + 1

    local_index_start = rank * local_size

else:

    local_size = chunk_size

    local_index_start = (
        remainder * (chunk_size + 1)
        + (rank - remainder) * chunk_size
    )


local_index_end = local_index_start + local_size


local_start = c_start + local_index_start
local_end = c_start + local_index_end - 1


print(f"Process {rank}/{size}: range [{local_start}, {local_end}]")


local_array = np.ones(local_size, dtype=np.int8)


comm.Barrier()

start_time = MPI.Wtime()


for p in divisors:

    first_multiple = (
        (local_start + p - 1) // p
    ) * p

    first_multiple = max(
        first_multiple,
        p * p
    )

    if first_multiple > local_end:
        continue

    for number in range(
        first_multiple,
        local_end + 1,
        p
    ):

        index = number - local_start

        local_array[index] = 0


local_primes = []

for i in range(local_size):

    if local_array[i] == 1:

        number = local_start + i

        local_primes.append(number)


gathered_primes = comm.gather(
    local_primes,
    root=0
)


end_time = MPI.Wtime()


if rank == 0:

    all_primes = divisors.copy()

    for process_primes in gathered_primes:

        all_primes.extend(process_primes)


    print("n =", n)
    print("Number of processes =", size)
    print("Number of prime numbers =", len(all_primes))
    print("Execution time =", end_time - start_time, "seconds")

    if n <= 1000:
        print("Prime numbers:")
        print(all_primes)