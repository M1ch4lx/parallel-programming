#!/usr/bin/env python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == 5, "Program wymaga dokładnie 5 procesów"

N = 25
chunk_size = N // size

# Alokacja współdzielonego okna pamięci (tylko proces 0 alokuje)
nbytes = N * np.dtype('i').itemsize if rank == 0 else 0
win = MPI.Win.Allocate_shared(nbytes, np.dtype('i').itemsize, comm=comm)

# zapytanie o wskaźnik (adres) do pamięci współdzielonej przypisanej do procesu o ranku 0 w oknie win
buf_ptr, itemsize = win.Shared_query(0)

#tworzy obiekt numpy ndarray, który korzysta z istniejącego bufora pamięci wskazywanego przez ten wskaźnik
shared_array = np.ndarray(buffer=buf_ptr, dtype='i', shape=(N,))


# Procesy zapisują swój numer w swojej części tablicy
start = rank * chunk_size
end = start + chunk_size
shared_array[start:end] = rank

win.Fence()  # synchronizacja zakończenia zapisu

if rank == 0:
    print("Zawartość współdzielonej tablicy:")
    print(shared_array)

win.Free()
