#include <errno.h>
#include <limits.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double element;

void transpose_rows(const element *a, element *b, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            b[j * n + i] = a[i * n + j];
}

void transpose_cols(const element *a, element *b, size_t n)
{
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            b[j * n + i] = a[i * n + j];
}

void transpose_blocked(const element *a, element *b, size_t n, size_t block)
{
    for (size_t ii = 0; ii < n; ii += block)
        for (size_t jj = 0; jj < n; jj += block) {
            size_t imax = ii + block < n ? ii + block : n;
            size_t jmax = jj + block < n ? jj + block : n;
            for (size_t i = ii; i < imax; ++i)
                for (size_t j = jj; j < jmax; ++j)
                    b[j * n + i] = a[i * n + j];
        }
}

void transpose_rows_omp(const element *a, element *b, size_t n, int *used)
{
    #pragma omp parallel default(none) shared(a, b, n, used)
    {
        #pragma omp single nowait
        *used = omp_get_num_threads();
        #pragma omp for schedule(runtime)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                b[j * n + i] = a[i * n + j];
    }
}

void transpose_blocked_omp(const element *a, element *b, size_t n, size_t block, int *used)
{
    size_t blocks = (n - 1) / block + 1;
    #pragma omp parallel default(none) shared(a, b, n, block, blocks, used)
    {
        #pragma omp single nowait
        *used = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (size_t bi = 0; bi < blocks; ++bi) {
            size_t ii = bi * block;
            size_t imax = ii + block < n ? ii + block : n;
            for (size_t jj = 0; jj < n; jj += block) {
                size_t jmax = jj + block < n ? jj + block : n;
                for (size_t i = ii; i < imax; ++i)
                    for (size_t j = jj; j < jmax; ++j)
                        b[j * n + i] = a[i * n + j];
            }
        }
    }
}

void transpose_inplace(element *a, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            element temp = a[i * n + j];
            a[i * n + j] = a[j * n + i];
            a[j * n + i] = temp;
        }
}

void transpose_inplace_omp(element *a, size_t n, int *used)
{
    #pragma omp parallel default(none) shared(a, n, used)
    {
        #pragma omp single nowait
        *used = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) {
                element temp = a[i * n + j];
                a[i * n + j] = a[j * n + i];
                a[j * n + i] = temp;
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
    if (argc < 5 || argc > 6 || !positive(argv[1], &n) ||
        !positive(argv[2], &block) || !positive(argv[3], &threads)) {
        fprintf(stderr, "Usage: %s N block threads version [static|static1|static8]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *names[] = {"rows", "cols", "blocked", "rows_omp", "blocked_omp", "inplace", "inplace_omp"};
    int version = -1;
    for (int i = 0; i < 7; ++i)
        if (!strcmp(argv[4], names[i])) version = i;
    const char *schedule = argc == 6 ? argv[5] : "static";
    int chunk = !strcmp(schedule, "static") ? 0 : !strcmp(schedule, "static1") ? 1 : !strcmp(schedule, "static8") ? 8 : -1;
    if (version < 0 || chunk < 0 || (argc == 6 && version != 3) ||
        n > SIZE_MAX / n / sizeof(element)) {
        fprintf(stderr, "Invalid version, schedule, or matrix size.\n");
        return EXIT_FAILURE;
    }
    size_t count = n * n, bytes = count * sizeof(element);
    element *a = malloc(bytes), *b = malloc(bytes), *reference = malloc(bytes);
    if (!a || !b || !reference) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(a); free(b); free(reference);
        return EXIT_FAILURE;
    }
    for (size_t k = 0; k < count; ++k) a[k] = (element)(k + 1);
    transpose_rows(a, reference, n);
    omp_set_dynamic(0);
    omp_set_num_threads((int)threads);
    omp_set_schedule(omp_sched_static, chunk);
    double times[5];
    size_t mismatches = 0;
    int used = 1;
    for (int run = 0; run < 6; ++run) {
        if (version >= 5) memcpy(b, a, bytes);
        else memset(b, 0, bytes);
        double start = omp_get_wtime();
        switch (version) {
            case 0: transpose_rows(a, b, n); break;
            case 1: transpose_cols(a, b, n); break;
            case 2: transpose_blocked(a, b, n, block); break;
            case 3: transpose_rows_omp(a, b, n, &used); break;
            case 4: transpose_blocked_omp(a, b, n, block, &used); break;
            case 5: transpose_inplace(b, n); break;
            case 6: transpose_inplace_omp(b, n, &used); break;
        }
        double elapsed = omp_get_wtime() - start;
        if (run > 0) times[run - 1] = elapsed;
        for (size_t k = 0; k < count; ++k)
            if (b[k] != reference[k]) ++mismatches;
    }
    qsort(times, 5, sizeof(double), compare_time);
    double bandwidth = times[2] > 0.0 ? 2.0 * (double)bytes / times[2] / 1e9 : 0.0;
    printf("N=%zu b=%zu threads=%d version=%s schedule=%s time=%.9f BW=%.6f mismatches=%zu check=%s repeats=5\n",
           n, block, used, names[version], schedule, times[2], bandwidth, mismatches, mismatches ? "FAIL" : "OK");
    printf("samples=%.9f,%.9f,%.9f,%.9f,%.9f\n", times[0], times[1], times[2], times[3], times[4]);
    free(a); free(b); free(reference);
    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
