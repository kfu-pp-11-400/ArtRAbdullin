#include <mpi.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void *allocate(size_t count, size_t element_size, int rank)
{
    void *buffer = NULL;
    if (count <= SIZE_MAX / element_size)
        buffer = malloc(count * element_size);
    if (buffer == NULL) {
        fprintf(stderr, "Rank %d: cannot allocate memory.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
    return buffer;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int n = 0;
    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Usage: %s N\n", argv[0]);
        } else {
            char *end = NULL;
            errno = 0;
            long long value = strtoll(argv[1], &end, 10);
            if (errno != 0 || end == argv[1] || *end != '\0' ||
                value <= 0 || value > INT_MAX) {
                fprintf(stderr, "N must be an integer from 1 to %d.\n", INT_MAX);
            } else if (value % processes != 0) {
                fprintf(stderr, "N must be divisible by the process count P=%d.\n", processes);
            } else {
                n = (int)value;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int local_n = n / processes;
    int *data = NULL;
    long long *local_sums = NULL;
    int *local_data = allocate((size_t)local_n, sizeof(*local_data), rank);

    if (rank == 0) {
        data = allocate((size_t)n, sizeof(*data), rank);
        local_sums = allocate((size_t)processes, sizeof(*local_sums), rank);
        for (int i = 0; i < n; ++i)
            data[i] = i + 1;
    }

    MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < local_n; ++i)
        local_sum += local_data[i];

    long long gather_sum = 0;
    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, local_sums, 1,
               MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        for (int r = 0; r < processes; ++r)
            gather_sum += local_sums[r];

    long long reduce_sum = 0;
    MPI_Reduce(&local_sum, &reduce_sum, 1, MPI_LONG_LONG_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    long long allreduce_sum = 0;
    MPI_Allreduce(&local_sum, &allreduce_sum, 1, MPI_LONG_LONG_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    long long expected = (long long)n * ((long long)n + 1) / 2;
    int valid = allreduce_sum == expected;
    if (rank == 0)
        valid = valid && gather_sum == expected && reduce_sum == expected;
    int all_valid = 0;
    MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N=%d P=%d local_n=%d\n", n, processes, local_n);
        for (int r = 0; r < processes; ++r)
            printf("Gather: rank=%d local_sum=%lld\n", r, local_sums[r]);
        printf("Expected=%lld\nGather=%lld\nReduce=%lld\nAllreduce=%lld\n",
               expected, gather_sum, reduce_sum, allreduce_sum);
        printf("Allreduce checked on %d processes\n", processes);
        printf("Time_seconds=%.9f\nCheck=%s\n", max_elapsed,
               all_valid ? "OK" : "FAIL");
    }

    free(local_sums);
    free(data);
    free(local_data);
    MPI_Finalize();
    return all_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
