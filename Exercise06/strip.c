#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define STRIP 64   // multiple of SIMD width: 4 doubles (AVX2) and 8 doubles (AVX-512)

int main(void) {
    double *A  = aligned_alloc(64, N * sizeof(double));
    double *B  = aligned_alloc(64, N * sizeof(double));
    double *C  = aligned_alloc(64, N * sizeof(double));
    double *Cs = aligned_alloc(64, N * sizeof(double));   // serial result for checking

    for (int i = 0; i < N; i++) { A[i] = i * 0.5; B[i] = (N - i) * 0.25; }

    double t0 = omp_get_wtime();
    for (int i = 0; i < N; i++) Cs[i] = A[i] * B[i];
    double ts = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < N; s += STRIP) {                  // threads share out the strips
        int end = (s + STRIP < N) ? s + STRIP : N;
        #pragma omp simd aligned(A, B, C : 64)
        for (int i = s; i < end; i++)                     // SIMD inside each strip
            C[i] = A[i] * B[i];
    }
    double tp = omp_get_wtime() - t0;

    int ok = 1;
    for (int i = 0; i < N; i++) if (C[i] != Cs[i]) { ok = 0; break; }
    printf("threads=%d serial=%.6f s parallel=%.6f s correct=%s\n",
           omp_get_max_threads(), ts, tp, ok ? "yes" : "NO");

    free(A); free(B); free(C); free(Cs);
    return 0;
}
