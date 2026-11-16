#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int parse_positive(const char *text, long long *value)
{
    if (*text == '\0') return 0;
    for (const char *p = text; *p; ++p)
        if (*p < '0' || *p > '9') return 0;
    errno = 0;
    char *end;
    *value = strtoll(text, &end, 10);
    return errno != ERANGE && *end == '\0' && *value > 0;
}

static double pi_sequential(long long n)
{
    double sum = 0.0;
    for (long long i = 0; i < n; ++i) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * (double)i + 1.0);
    }
    return 4.0 * sum;
}

static double pi_parallel(long long n, int *used_threads)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static) default(none) shared(n, used_threads)
    for (long long i = 0; i < n; ++i) {
        if (i == 0) *used_threads = omp_get_num_threads();
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * (double)i + 1.0);
    }
    return 4.0 * sum;
}

static void print_result(const char *mode, long long n, int threads,
                         double pi, double seconds)
{
    printf("mode=%s n=%lld threads=%d pi=%.17g abs_error=%.12e seconds=%.9f\n",
           mode, n, threads, pi, fabs(pi - M_PI), seconds);
}

int main(int argc, char **argv)
{
    long long n, requested_threads;
    if (argc < 3 || argc > 4 || !parse_positive(argv[1], &n) ||
        !parse_positive(argv[2], &requested_threads) || requested_threads > INT_MAX) {
        fprintf(stderr, "Usage: %s n threads [both|seq|par]\n"
                        "n and threads must be positive decimal integers.\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *mode = argc == 4 ? argv[3] : "both";
    if (strcmp(mode, "both") && strcmp(mode, "seq") && strcmp(mode, "par")) {
        fprintf(stderr, "Mode must be both, seq or par.\n");
        return EXIT_FAILURE;
    }

    omp_set_dynamic(0);
    omp_set_num_threads((int)requested_threads);
    double seq_seconds = 0.0, par_seconds = 0.0;
    if (strcmp(mode, "par") != 0) {
        double start = omp_get_wtime();
        double pi = pi_sequential(n);
        seq_seconds = omp_get_wtime() - start;
        print_result("seq", n, 1, pi, seq_seconds);
    }
    if (strcmp(mode, "seq") != 0) {
        int used_threads = 0;
        double start = omp_get_wtime();
        double pi = pi_parallel(n, &used_threads);
        par_seconds = omp_get_wtime() - start;
        print_result("par", n, used_threads, pi, par_seconds);
    }
    if (strcmp(mode, "both") == 0 && par_seconds > 0.0)
        printf("speedup=%.6f\n", seq_seconds / par_seconds);
    return EXIT_SUCCESS;
}
