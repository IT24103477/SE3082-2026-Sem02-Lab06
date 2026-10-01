#include <omp.h>
#include <stdio.h>

#define NPOINTS 2000
#define MAXITER 2000

struct complex { double real; double imag; };

int main(void) {
    int numoutside = 0;
    double area, error;
    double t0 = omp_get_wtime();

    #pragma omp parallel reduction(+:numoutside)
    {
        int tid      = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int chunk    = (NPOINTS + nthreads - 1) / nthreads;   // equal share of rows
        int start    = tid * chunk;
        int end      = (start + chunk < NPOINTS) ? start + chunk : NPOINTS;

        for (int i = start; i < end; i++) {
            for (int j = 0; j < NPOINTS; j++) {
                struct complex z, c;      // declared inside, so private to each thread
                double ztemp;
                c.real = -2.0 + 2.5 * (double)i / (double)NPOINTS + 1.0e-7;
                c.imag = 1.125 * (double)j / (double)NPOINTS + 1.0e-7;
                z = c;
                for (int iter = 0; iter < MAXITER; iter++) {
                    ztemp  = (z.real * z.real) - (z.imag * z.imag) + c.real;
                    z.imag = z.real * z.imag * 2 + c.imag;
                    z.real = ztemp;
                    if ((z.real * z.real + z.imag * z.imag) > 4.0e0) { numoutside++; break; }
                }
            }
        }
    }

    double t = omp_get_wtime() - t0;
    area  = 2.0 * 2.5 * 1.125 * (double)(NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
    error = area / (double)NPOINTS;
    printf("threads=%d Area = %12.8f +/- %12.8f  time=%.3f s\n", omp_get_max_threads(), area, error, t);
    return 0;
}
