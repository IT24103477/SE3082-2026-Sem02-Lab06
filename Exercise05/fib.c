#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define CUTOFF 20

int fib_serial(int n) {
    if (n < 2) return n;
    return fib_serial(n - 1) + fib_serial(n - 2);
}

int fib(int n) {
    int i, j;
    if (n < 2) return n;
    if (n < CUTOFF) return fib_serial(n);   // small problems: no tasks

    #pragma omp task shared(i)
    i = fib(n - 1);
    #pragma omp task shared(j)
    j = fib(n - 2);
    #pragma omp taskwait                    // wait for both children
    return i + j;
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 40;
    int result;

    double t0 = omp_get_wtime();
    int s = fib_serial(n);
    double tserial = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        result = fib(n);
    }
    double tpar = omp_get_wtime() - t0;

    printf("n=%d serial=%d (%.3f s) parallel=%d (%.3f s) threads=%d\n",
           n, s, tserial, result, tpar, omp_get_max_threads());
    return 0;
}
