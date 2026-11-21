#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { SERIAL, OMP, SIMD, OMP_SIMD } implementation;

typedef struct {
    size_t n, d;
    double *x, *y;
} dataset;

typedef struct {
    double *w, b, loss;
} model;

static uint64_t rng_state = UINT64_C(0x4d595df4d0f33173);

static double random_value(void)
{
    rng_state = rng_state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return (double)(rng_state >> 11) * 0x1.0p-53 - 0.5;
}

static double soft_threshold(double z, double alpha)
{
    if (z > alpha) return z - alpha;
    if (z < -alpha) return z + alpha;
    return 0.0;
}

static const char *implementation_name(implementation kind)
{
    static const char *names[] = {"serial", "omp", "simd", "omp_simd"};
    return names[kind];
}

static int parse_size(const char *text, size_t *value)
{
    if (*text == '\0') return 0;
    for (const char *p = text; *p; ++p)
        if (*p < '0' || *p > '9') return 0;
    errno = 0;
    char *end;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *end || !parsed || parsed > SIZE_MAX) return 0;
    *value = (size_t)parsed;
    return 1;
}

static int parse_double(const char *text, double *value)
{
    errno = 0;
    char *end;
    *value = strtod(text, &end);
    return end != text && *end == '\0' && errno != ERANGE && isfinite(*value);
}

static int allocate_dataset(dataset *data, size_t n, size_t d)
{
    if (n > SIZE_MAX / d || n * d > SIZE_MAX / sizeof(double)) return 0;
    data->n = n;
    data->d = d;
    data->x = malloc(n * d * sizeof(*data->x));
    data->y = malloc(n * sizeof(*data->y));
    return data->x && data->y;
}

static void generate_dataset(dataset *data)
{
    rng_state = UINT64_C(0x4d595df4d0f33173);
    for (size_t i = 0; i < data->n; ++i) {
        double target = 0.2;
        for (size_t j = 0; j < data->d; ++j) {
            double value = random_value();
            data->x[i * data->d + j] = value;
            if (j % 7 == 0) target += (0.3 + 0.1 * (double)(j % 5)) * value;
        }
        data->y[i] = target + 0.02 * random_value();
    }
}

static int allocate_model(model *result, size_t d)
{
    result->w = calloc(d, sizeof(*result->w));
    result->b = 0.0;
    result->loss = 0.0;
    return result->w != NULL;
}

static void compute_predictions(const dataset *data, const model *current,
                                double *errors, implementation kind, int threads)
{
    if (kind == OMP || kind == OMP_SIMD) {
        /* Objects are independent; each OpenMP iteration writes one error. */
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t i = 0; i < data->n; ++i) {
            double prediction = current->b;
            if (kind == OMP_SIMD) {
                /* The row is contiguous, so this reduction is suitable for SIMD. */
                #pragma omp simd reduction(+:prediction)
                for (size_t j = 0; j < data->d; ++j)
                    prediction += data->x[i * data->d + j] * current->w[j];
            } else {
                for (size_t j = 0; j < data->d; ++j)
                    prediction += data->x[i * data->d + j] * current->w[j];
            }
            errors[i] = prediction - data->y[i];
        }
    } else {
        for (size_t i = 0; i < data->n; ++i) {
            double prediction = current->b;
            if (kind == SIMD) {
                #pragma omp simd reduction(+:prediction)
                for (size_t j = 0; j < data->d; ++j)
                    prediction += data->x[i * data->d + j] * current->w[j];
            } else {
                for (size_t j = 0; j < data->d; ++j)
                    prediction += data->x[i * data->d + j] * current->w[j];
            }
            errors[i] = prediction - data->y[i];
        }
    }
}

static void compute_gradients(const dataset *data, const double *errors, double *gradients,
                              double *grad_b, implementation kind, int threads,
                              double *partial, double *partial_b)
{
    if (kind == OMP || kind == OMP_SIMD) {
        memset(partial, 0, (size_t)threads * data->d * sizeof(*partial));
        /* Every thread owns one row of partial gradients: no race on gradients. */
        #pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            double *local = partial + (size_t)tid * data->d;
            double local_b = 0.0;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < data->n; ++i) {
                double error = errors[i];
                local_b += error;
                if (kind == OMP_SIMD) {
                    /* Values in a feature row are contiguous; independent updates to local[j]. */
                    #pragma omp simd
                    for (size_t j = 0; j < data->d; ++j)
                        local[j] += error * data->x[i * data->d + j];
                } else {
                    for (size_t j = 0; j < data->d; ++j)
                        local[j] += error * data->x[i * data->d + j];
                }
            }
            partial_b[tid] = local_b;
        }
        memset(gradients, 0, data->d * sizeof(*gradients));
        *grad_b = 0.0;
        for (int t = 0; t < threads; ++t) {
            *grad_b += partial_b[t];
            for (size_t j = 0; j < data->d; ++j)
                gradients[j] += partial[(size_t)t * data->d + j];
        }
    } else {
        memset(gradients, 0, data->d * sizeof(*gradients));
        *grad_b = 0.0;
        for (size_t i = 0; i < data->n; ++i) {
            double error = errors[i];
            *grad_b += error;
            if (kind == SIMD) {
                #pragma omp simd
                for (size_t j = 0; j < data->d; ++j)
                    gradients[j] += error * data->x[i * data->d + j];
            } else {
                for (size_t j = 0; j < data->d; ++j)
                    gradients[j] += error * data->x[i * data->d + j];
            }
        }
    }
    for (size_t j = 0; j < data->d; ++j)
        gradients[j] /= (double)data->n;
    *grad_b /= (double)data->n;
}

static void train(const dataset *data, model *result, double lambda, double rate,
                  size_t iterations, implementation kind, int threads)
{
    double *errors = malloc(data->n * sizeof(*errors));
    double *gradients = malloc(data->d * sizeof(*gradients));
    double *partial = NULL, *partial_b = NULL;
    if (kind == OMP || kind == OMP_SIMD) {
        partial = malloc((size_t)threads * data->d * sizeof(*partial));
        partial_b = malloc((size_t)threads * sizeof(*partial_b));
    }
    if (!errors || !gradients || ((kind == OMP || kind == OMP_SIMD) && (!partial || !partial_b))) {
        fprintf(stderr, "Training workspace allocation failed.\n");
        free(errors); free(gradients); free(partial); free(partial_b); exit(EXIT_FAILURE);
    }
    for (size_t step = 0; step < iterations; ++step) {
        double grad_b;
        compute_predictions(data, result, errors, kind, threads);
        compute_gradients(data, errors, gradients, &grad_b, kind, threads, partial, partial_b);
        for (size_t j = 0; j < data->d; ++j)
            result->w[j] = soft_threshold(result->w[j] - rate * gradients[j], rate * lambda);
        result->b -= rate * grad_b;
    }
    compute_predictions(data, result, errors, kind, threads);
    double squared = 0.0, l1 = 0.0;
    for (size_t i = 0; i < data->n; ++i) squared += errors[i] * errors[i];
    for (size_t j = 0; j < data->d; ++j) l1 += fabs(result->w[j]);
    result->loss = squared / (2.0 * (double)data->n) + lambda * l1;
    free(errors); free(gradients); free(partial); free(partial_b);
}

int main(int argc, char **argv)
{
    if (argc != 8) {
        fprintf(stderr, "Usage: %s N d lambda rate iterations threads {serial|omp|simd|omp_simd}\n", argv[0]);
        return EXIT_FAILURE;
    }
    size_t n, d, iterations, threads;
    double lambda, rate;
    if (!parse_size(argv[1], &n) || !parse_size(argv[2], &d) || !parse_double(argv[3], &lambda) ||
        !parse_double(argv[4], &rate) || !parse_size(argv[5], &iterations) || !parse_size(argv[6], &threads) ||
        lambda < 0.0 || rate <= 0.0 || threads > INT_MAX) {
        fprintf(stderr, "Invalid numeric parameter.\n");
        return EXIT_FAILURE;
    }
    implementation kind;
    if (!strcmp(argv[7], "serial")) kind = SERIAL;
    else if (!strcmp(argv[7], "omp")) kind = OMP;
    else if (!strcmp(argv[7], "simd")) kind = SIMD;
    else if (!strcmp(argv[7], "omp_simd")) kind = OMP_SIMD;
    else { fprintf(stderr, "Unknown implementation.\n"); return EXIT_FAILURE; }
    dataset data = {0}; model result = {0};
    if (!allocate_dataset(&data, n, d) || !allocate_model(&result, d)) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(data.x); free(data.y); free(result.w); return EXIT_FAILURE;
    }
    generate_dataset(&data);
    omp_set_dynamic(0);
    double start = omp_get_wtime();
    train(&data, &result, lambda, rate, iterations, kind, (int)threads);
    double seconds = omp_get_wtime() - start;
    size_t zeros = 0;
    for (size_t j = 0; j < d; ++j) if (fabs(result.w[j]) < 1e-6) ++zeros;
    size_t actual_threads = (kind == OMP || kind == OMP_SIMD) ? threads : 1;
    printf("implementation=%s N=%zu d=%zu lambda=%.12g rate=%.12g iterations=%zu threads=%zu loss=%.17g zeros=%zu time=%.9f\n",
           implementation_name(kind), n, d, lambda, rate, iterations, actual_threads, result.loss, zeros, seconds);
    free(data.x); free(data.y); free(result.w);
    return EXIT_SUCCESS;
}
