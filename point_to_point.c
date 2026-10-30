#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TAG_MESSAGE = 100,
    TAG_REPLY = 101,
    TAG_DATA = 200,
    TAG_ECHO = 201,
    TEXT_CAPACITY = 128
};

static void fail(int rank, const char *message)
{
    fprintf(stderr, "[rank %d] %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

static void check_received(int rank, const MPI_Status *status,
                           int source, int tag, int expected_count)
{
    int count;
    MPI_Get_count(status, MPI_CHAR, &count);
    if (status->MPI_SOURCE != source || status->MPI_TAG != tag || count != expected_count)
        fail(rank, "Unexpected source, tag or message length.");
}

static void exchange_messages(int rank, int size)
{
    char message[TEXT_CAPACITY], reply[TEXT_CAPACITY], expected[TEXT_CAPACITY];
    MPI_Status status;
    if (rank == 0) {
        for (int destination = 1; destination < size; ++destination) {
            int length = snprintf(message, sizeof(message), "Hello from rank 0 to rank %d", destination);
            printf("[rank 0] sending to rank %d, tag=%d: %s\n", destination, TAG_MESSAGE, message);
            MPI_Send(message, length + 1, MPI_CHAR, destination, TAG_MESSAGE, MPI_COMM_WORLD);
        }
        int replies = 0;
        for (int source = 1; source < size; ++source) {
            memset(reply, 0, sizeof(reply));
            MPI_Recv(reply, TEXT_CAPACITY, MPI_CHAR, source, TAG_REPLY, MPI_COMM_WORLD, &status);
            int length = snprintf(expected, sizeof(expected), "Message received by rank %d", source);
            check_received(rank, &status, source, TAG_REPLY, length + 1);
            if (memcmp(reply, expected, (size_t)length + 1) != 0)
                fail(rank, "Incorrect reply text.");
            printf("[rank 0] received from rank %d, tag=%d: %s\n", source, status.MPI_TAG, reply);
            ++replies;
        }
        printf("SUMMARY processes=%d sent=%d replies=%d check=OK\n", size, size - 1, replies);
    } else {
        memset(message, 0, sizeof(message));
        MPI_Recv(message, TEXT_CAPACITY, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        int length = snprintf(expected, sizeof(expected), "Hello from rank 0 to rank %d", rank);
        check_received(rank, &status, 0, TAG_MESSAGE, length + 1);
        if (memcmp(message, expected, (size_t)length + 1) != 0)
            fail(rank, "Message is addressed to a different rank.");
        printf("[rank %d] received from rank 0, tag=%d: %s\n", rank, status.MPI_TAG, message);
        length = snprintf(reply, sizeof(reply), "Message received by rank %d", rank);
        printf("[rank %d] sending reply to rank 0, tag=%d\n", rank, TAG_REPLY);
        MPI_Send(reply, length + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

static int parse_number(const char *text, int allow_zero, int *number)
{
    if (!*text) return 0;
    for (const char *p = text; *p; ++p)
        if (*p < '0' || *p > '9') return 0;
    errno = 0;
    char *end;
    long value = strtol(text, &end, 10);
    if (errno || *end || value < (allow_zero ? 0 : 1) || value > INT_MAX) return 0;
    *number = (int)value;
    return 1;
}

static void benchmark(int rank, int repeats, int warmup)
{
    const int sizes[] = {1, 1024, 65536, 1048576};
    size_t capacity = (size_t)sizes[3];
    char *data = malloc(capacity);
    char *echo = rank == 0 ? malloc(capacity) : NULL;
    if (!data || (rank == 0 && !echo)) fail(rank, "Buffer allocation failed.");
    for (size_t i = 0; i < capacity; ++i) data[i] = (char)('a' + i % 26);

    for (int s = 0; s < 4; ++s) {
        int bytes = sizes[s];
        double total = 0.0;
        for (int stage = 0; stage < 2; ++stage) {
            int count = stage == 0 ? warmup : repeats;
            for (int iteration = 0; iteration < count; ++iteration) {
                MPI_Status status;
                if (rank == 0) {
                    double start = MPI_Wtime();
                    MPI_Send(data, bytes, MPI_CHAR, 1, TAG_DATA, MPI_COMM_WORLD);
                    MPI_Recv(echo, bytes, MPI_CHAR, 1, TAG_ECHO, MPI_COMM_WORLD, &status);
                    double elapsed = MPI_Wtime() - start;
                    if (stage == 1) total += elapsed;
                    check_received(rank, &status, 1, TAG_ECHO, bytes);
                    if (memcmp(data, echo, (size_t)bytes) != 0)
                        fail(rank, "Echo data mismatch.");
                } else {
                    MPI_Recv(data, bytes, MPI_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, &status);
                    check_received(rank, &status, 0, TAG_DATA, bytes);
                    MPI_Send(data, bytes, MPI_CHAR, 0, TAG_ECHO, MPI_COMM_WORLD);
                }
            }
        }
        if (rank == 0)
            printf("BENCH bytes=%d repeats=%d warmup=%d mean_rtt_seconds=%.12f check=OK\n",
                   bytes, repeats, warmup, total / (double)repeats);
    }
    free(data);
    free(echo);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    setvbuf(stdout, NULL, _IONBF, 0);
    int repeats = 1000, warmup = 20;
    int bench = argc >= 2 && strcmp(argv[1], "bench") == 0;
    int valid = argc == 1 ||
        (bench && argc <= 4 && (argc < 3 || parse_number(argv[2], 0, &repeats)) &&
         (argc < 4 || parse_number(argv[3], 1, &warmup)));
    if (!valid || size < 2 || (bench && size != 2)) {
        if (rank == 0) {
            if (size < 2) fprintf(stderr, "At least two MPI processes are required.\n");
            else if (bench && size != 2) fprintf(stderr, "Benchmark requires exactly two MPI processes.\n");
            else fprintf(stderr, "Usage: %s [bench [repeats [warmup]]]\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (bench) benchmark(rank, repeats, warmup);
    else exchange_messages(rank, size);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
