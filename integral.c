#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double f(double x)
{
    return cos(x * x);
}

static int positive_integer(const char *text, long long *value)
{
    if (*text == '\0') return 0;
    for (const char *p = text; *p; ++p)
        if (*p < '0' || *p > '9') return 0;
    errno = 0;
    char *end;
    *value = strtoll(text, &end, 10);
    return errno != ERANGE && *end == '\0' && *value > 0;
}

static int real_number(const char *text, double *value)
{
    errno = 0;
    char *end;
    *value = strtod(text, &end);
    return end != text && *end == '\0' && errno != ERANGE && isfinite(*value);
}

static double sequential(double a, double b, long long n)
{
    double h = (b - a) / (double)n;
    double sum = 0.0;
    for (long long i = 0; i < n; ++i) {
        double x = a + ((double)i + 0.5) * h;
        sum += f(x);
    }
    return h * sum;
}

static double parallel(double a, double b, long long n, int *threads)
{
    double h = (b - a) / (double)n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static) default(none) shared(a, h, n, threads)
    for (long long i = 0; i < n; ++i) {
        if (i == 0) *threads = omp_get_num_threads();
        double x = a + ((double)i + 0.5) * h;
        sum += f(x);
    }
    return h * sum;
}

static double integrate(double a, double b, long long n, int is_parallel, int *threads)
{
    if (is_parallel) return parallel(a, b, n, threads);
    *threads = 1;
    return sequential(a, b, n);
}

static int measure(double a, double b, long long n, double epsilon,
                   int is_parallel, double *seconds)
{
    int threads = 1;
    double start = omp_get_wtime();
    double value = integrate(a, b, n, is_parallel, &threads);
    double estimate = 0.0;
    int refinements = 0;
    if (!isfinite(value)) {
        fprintf(stderr, "Non-finite integral: check the interval and function domain.\n");
        return 0;
    }
    if (epsilon > 0.0) {
        do {
            if (n > LLONG_MAX / 2 || refinements >= 30) {
                fprintf(stderr, "Refinement limit reached; convergence not established.\n");
                return 0;
            }
            double previous = value;
            n *= 2;
            value = integrate(a, b, n, is_parallel, &threads);
            if (!isfinite(value)) {
                fprintf(stderr, "Non-finite integral during refinement.\n");
                return 0;
            }
            estimate = fabs(value - previous) / 3.0;
            ++refinements;
        } while (estimate > epsilon);
    }
    *seconds = omp_get_wtime() - start;
    printf("mode=%s n=%lld threads=%d integral=%.17g seconds=%.9f",
           is_parallel ? "par" : "seq", n, threads, value, *seconds);
    if (epsilon > 0.0)
        printf(" epsilon=%.12e estimate=%.12e refinements=%d", epsilon, estimate, refinements);
    printf("\n");
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "Usage:\n  %s fixed a b n threads [both|seq|par]\n"
                        "  %s adaptive a b n0 threads epsilon [both|seq|par]\n", argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    int adaptive = strcmp(argv[1], "adaptive") == 0;
    int required = adaptive ? 7 : 6;
    if ((!adaptive && strcmp(argv[1], "fixed") != 0) || argc < required || argc > required + 1) {
        fprintf(stderr, "Invalid mode or argument count.\n");
        return EXIT_FAILURE;
    }
    double a, b, epsilon = 0.0;
    long long n, requested_threads;
    if (!real_number(argv[2], &a) || !real_number(argv[3], &b) ||
        !isfinite(b - a) || !positive_integer(argv[4], &n) ||
        !positive_integer(argv[5], &requested_threads) || requested_threads > INT_MAX ||
        (adaptive && (!real_number(argv[6], &epsilon) || epsilon <= 0.0))) {
        fprintf(stderr, "Invalid numeric argument.\n");
        return EXIT_FAILURE;
    }
    const char *mode = argc == required + 1 ? argv[required] : "both";
    if (strcmp(mode, "both") && strcmp(mode, "seq") && strcmp(mode, "par")) {
        fprintf(stderr, "Execution mode must be both, seq or par.\n");
        return EXIT_FAILURE;
    }
    omp_set_dynamic(0);
    omp_set_num_threads((int)requested_threads);
    double seq_seconds = 0.0, par_seconds = 0.0;
    if (strcmp(mode, "par") != 0 && !measure(a, b, n, epsilon, 0, &seq_seconds))
        return EXIT_FAILURE;
    if (strcmp(mode, "seq") != 0 && !measure(a, b, n, epsilon, 1, &par_seconds))
        return EXIT_FAILURE;
    if (strcmp(mode, "both") == 0 && par_seconds > 0.0)
        printf("speedup=%.6f\n", seq_seconds / par_seconds);
    return EXIT_SUCCESS;
}
