# Rownolegly Branch and Bound — TSP na okregu

Instancja: N=13, seed=2026, promien=100.0, bound=minedge.
Czas T jest czasem zewnetrznym subprocess, tzn. obejmuje uruchomienie MPI,
przygotowanie danych, generowanie wszystkich prefiksow, obliczenia i zamkniecie MPI.
T(p) jest mediana z powtorzen. p oznacza workery, poza nimi istnieje proces master.

## Eksperyment 1 — stale d, zmienna liczba workerow

| d | p | taski | T(p) [s] | S(p) | E(p) | serial fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 132 | 12.6229 | 1.0000 | 1.0000 | — |
| 3 | 2 | 132 | 6.7207 | 1.8782 | 0.9391 | 0.0649 |
| 3 | 4 | 132 | 3.7501 | 3.3660 | 0.8415 | 0.0628 |
| 3 | 8 | 132 | 2.3340 | 5.4083 | 0.6760 | 0.0685 |

## Eksperyment 2 — stale p, zmienna glebokosc d

| d | p | taski | T1(d) [s] | Tp(d) [s] | S(p,d) | E(p,d) | serial fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 12 | 3.8244 | 1.3170 | 2.9038 | 0.3630 | 0.2507 |
| 3 | 8 | 132 | 12.6229 | 2.3340 | 5.4083 | 0.6760 | 0.0685 |
| 4 | 8 | 1320 | 42.8080 | 5.9770 | 7.1621 | 0.8953 | 0.0167 |

## Wzory i interpretacja

- Przyspieszenie S(p,d) = T(1,d) / T(p,d).
- Efektywnosc E(p,d) = S(p,d) / p.
- Serial fraction f(p,d) = (1/S(p,d) - 1/p) / (1 - 1/p), tylko p > 1.
- Glebokosc d zmienia nie tylko liczbe taskow, ale i skutecznosc lokalnego pruningu;
  T(1,d) moze byc rozne dla roznych d. Z tego powodu porownujemy z bazowym T(1,d).
- Ujemna serial fraction moze wystapic przy przyspieszeniu superliniowym lub szumie pomiarowym.
- Dla p=1 serial fraction jest nieokreslona (myslnik w tabeli).

## PCAM

1. Partitioning: wszystkie prefiksy od miasta 1 do glebokosci d.
2. Communication: master przesyla prefiks; worker odsyla koszt i trase;
   brak wymiany lokalnych gornych granic miedzy workerami.
3. Agglomeration: jeden task = cale poddrzewo jednego prefiksu; chunksize=1.
4. Mapping: dynamiczny przydzial wolnym workerom przez MPIPoolExecutor.map.

## Pliki

measurements.csv — surowe pomiary; summary.csv — mediany i metryki;
exp1_*.png i exp2_*.png — szesc wymaganych wykresow;
runs/*.json — wyniki TSP; logs/*.log — logi uruchomien MPI.
