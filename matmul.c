#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ijk_seq(const double *a, const double *b, double *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

void ijk_omp(const double *a, const double *b, double *c, size_t n, int *used)
{
    #pragma omp parallel default(none) shared(a, b, c, n, used)
    {
        #pragma omp single nowait
        *used = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k)
                    sum += a[i * n + k] * b[k * n + j];
                c[i * n + j] = sum;
            }
    }
}

void ikj_seq(const double *a, const double *b, double *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n; ++k) {
            double value = a[i * n + k];
            for (size_t j = 0; j < n; ++j)
                c[i * n + j] += value * b[k * n + j];
        }
}

void ikj_omp(const double *a, const double *b, double *c, size_t n, int *used)
{
    #pragma omp parallel default(none) shared(a, b, c, n, used)
    {
        #pragma omp single nowait
        *used = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < n; ++k) {
                double value = a[i * n + k];
                for (size_t j = 0; j < n; ++j)
                    c[i * n + j] += value * b[k * n + j];
            }
    }
}

void blocked_omp(const double *a, const double *b, double *c, size_t n, size_t block, int *used)
{
    size_t blocks = (n - 1) / block + 1;
    #pragma omp parallel default(none) shared(a, b, c, n, block, blocks, used)
    {
        #pragma omp single nowait
        *used = omp_get_num_threads();
        #pragma omp for collapse(2) schedule(static)
        for (size_t bi = 0; bi < blocks; ++bi)
            for (size_t bj = 0; bj < blocks; ++bj) {
                size_t ii = bi * block, jj = bj * block;
                size_t imax = ii + block < n ? ii + block : n;
                size_t jmax = jj + block < n ? jj + block : n;
                for (size_t kk = 0; kk < n; kk += block) {
                    size_t kmax = kk + block < n ? kk + block : n;
                    for (size_t i = ii; i < imax; ++i)
                        for (size_t k = kk; k < kmax; ++k) {
                            double value = a[i * n + k];
                            for (size_t j = jj; j < jmax; ++j)
                                c[i * n + j] += value * b[k * n + j];
                        }
                }
            }
    }
}

static int positive(const char *s, size_t *value)
{
    if (!*s) return 0;
    for (const char *p = s; *p; ++p)
        if (*p < '0' || *p > '9') return 0;
    errno = 0;
    char *end;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || *end || !v || v > INT_MAX) return 0;
    *value = (size_t)v;
    return 1;
}

static int compare_time(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    size_t n, block, threads;
    if (argc != 5 || !positive(argv[1], &n) || !positive(argv[2], &block) || !positive(argv[3], &threads)) {
        fprintf(stderr, "Usage: %s N block threads {ijk_seq|ijk_omp|ikj_seq|ikj_omp|blocked_omp}\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *names[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};
    int version = -1;
    for (int i = 0; i < 5; ++i)
        if (!strcmp(argv[4], names[i])) version = i;
    if (version < 0 || n > SIZE_MAX / n / sizeof(double)) {
        fprintf(stderr, "Invalid version or matrix size.\n");
        return EXIT_FAILURE;
    }
    size_t count = n * n, bytes = count * sizeof(double);
    double *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes), *reference = malloc(bytes);
    if (!a || !b || !c || !reference) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(a); free(b); free(c); free(reference);
        return EXIT_FAILURE;
    }
    for (size_t k = 0; k < count; ++k) {
        a[k] = ((int)((k % 101) * 17 % 101) - 50) / 51.0;
        b[k] = ((int)((k % 97) * 13 % 97) - 48) / 49.0;
    }
    memset(reference, 0, bytes);
    ijk_seq(a, b, reference, n);
    omp_set_dynamic(0);
    omp_set_num_threads((int)threads);
    double times[5], max_error = 0.0;
    size_t mismatches = 0;
    int used = 1;
    for (int run = 0; run < 6; ++run) {
        memset(c, 0, bytes);
        double start = omp_get_wtime();
        switch (version) {
            case 0: ijk_seq(a, b, c, n); break;
            case 1: ijk_omp(a, b, c, n, &used); break;
            case 2: ikj_seq(a, b, c, n); break;
            case 3: ikj_omp(a, b, c, n, &used); break;
            case 4: blocked_omp(a, b, c, n, block, &used); break;
        }
        double elapsed = omp_get_wtime() - start;
        if (run > 0) times[run - 1] = elapsed;
        for (size_t k = 0; k < count; ++k) {
            double error = fabs(c[k] - reference[k]);
            if (error > max_error) max_error = error;
            if (!isfinite(c[k]) || error > 1e-9 * (1.0 + fabs(reference[k]))) ++mismatches;
        }
    }
    qsort(times, 5, sizeof(double), compare_time);
    printf("N=%zu b=%zu threads=%d version=%s time=%.9f max_error=%.12e mismatches=%zu check=%s repeats=5\n",
           n, block, used, names[version], times[2], max_error, mismatches, mismatches ? "FAIL" : "OK");
    printf("samples=%.9f,%.9f,%.9f,%.9f,%.9f\n", times[0], times[1], times[2], times[3], times[4]);
    free(a); free(b); free(c); free(reference);
    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
