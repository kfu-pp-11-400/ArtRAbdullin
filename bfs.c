#include <mpi.h>
#include <omp.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n, m, arcs;
    int *offsets, *edges;
} Graph;

typedef struct {
    int *data;
    size_t size, capacity;
    int failed;
    long long scanned;
} List;

typedef struct {
    int *frontier, *next, *local, *global, *sizes;
    List *lists;
    int max_threads, actual_threads, levels;
    long long processed, scanned;
} Workspace;

static void fail(const char *message)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

static void *allocate(size_t count, size_t item_size)
{
    if (count == 0)
        count = 1;
    if (count > SIZE_MAX / item_size)
        fail("Allocation size overflow.");
    void *p = calloc(count, item_size);
    if (!p)
        fail("Cannot allocate memory.");
    return p;
}

static int parse_number(const char *text, int minimum, int maximum)
{
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < minimum || value > maximum)
        fail("Invalid numeric argument.");
    return (int)value;
}

static Graph read_graph(const char *path, int directed)
{
    Graph g = {0};
    FILE *file = fopen(path, "r");
    if (!file)
        fail("Cannot open graph file.");
    long long n, m;
    if (fscanf(file, "%lld %lld", &n, &m) != 2 || n < 1 || n >= INT_MAX ||
        m < 0 || m > INT_MAX / (directed ? 1 : 2))
        fail("Invalid graph header: expected N > 0 and a supported edge count.");
    g.n = (int)n;
    g.m = (int)m;
    g.arcs = g.m * (directed ? 1 : 2);
    g.offsets = allocate((size_t)g.n + 1, sizeof(int));
    g.edges = allocate((size_t)g.arcs, sizeof(int));
    int *from = allocate((size_t)g.m, sizeof(int));
    int *to = allocate((size_t)g.m, sizeof(int));
    for (int i = 0; i < g.m; ++i) {
        long long u, v;
        if (fscanf(file, "%lld %lld", &u, &v) != 2 ||
            u < 0 || v < 0 || u >= g.n || v >= g.n)
            fail("Invalid edge or incomplete graph file.");
        from[i] = (int)u;
        to[i] = (int)v;
        ++g.offsets[u + 1];
        if (!directed)
            ++g.offsets[v + 1];
    }
    char extra;
    if (fscanf(file, " %c", &extra) == 1)
        fail("Unexpected data after the edge list.");
    fclose(file);
    for (int i = 1; i <= g.n; ++i)
        g.offsets[i] += g.offsets[i - 1];
    int *position = allocate((size_t)g.n, sizeof(int));
    memcpy(position, g.offsets, (size_t)g.n * sizeof(int));
    for (int i = 0; i < g.m; ++i) {
        g.edges[position[from[i]]++] = to[i];
        if (!directed)
            g.edges[position[to[i]]++] = from[i];
    }
    free(position);
    free(from);
    free(to);
    return g;
}

static void sequential_bfs(const Graph *g, int source, int *dist, int *queue)
{
    for (int i = 0; i < g->n; ++i)
        dist[i] = -1;
    int head = 0, tail = 0;
    queue[tail++] = source;
    dist[source] = 0;
    while (head < tail) {
        int v = queue[head++];
        for (int j = g->offsets[v]; j < g->offsets[v + 1]; ++j) {
            int u = g->edges[j];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
}

static int append(List *list, int vertex)
{
    if (list->size == list->capacity) {
        if (list->capacity > SIZE_MAX / (2 * sizeof(int))) {
            list->failed = 1;
            return 0;
        }
        size_t capacity = list->capacity ? list->capacity * 2 : 64;
        int *data = realloc(list->data, capacity * sizeof(int));
        if (!data) {
            list->failed = 1;
            return 0;
        }
        list->data = data;
        list->capacity = capacity;
    }
    list->data[list->size++] = vertex;
    return 1;
}

static void hybrid_bfs(const Graph *g, int source, int *dist, Workspace *w,
                       int rank, int processes)
{
    for (int i = 0; i < g->n; ++i)
        dist[i] = -1;
    dist[source] = 0;
    int size = 1;
    w->frontier[0] = source;
    w->levels = 0;
    w->processed = 0;
    w->scanned = 0;
    while (size > 0) {
        int level = w->levels;
        w->sizes[w->levels++] = size;
        memset(w->local, 0, (size_t)g->n * sizeof(int));
        for (int t = 0; t < w->max_threads; ++t) {
            w->lists[t].size = 0;
            w->lists[t].scanned = 0;
        }
        int local_size = size / processes + (rank < size % processes);
        w->processed += local_size;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            List *list = &w->lists[tid];
            #pragma omp single
            w->actual_threads = omp_get_num_threads();
            #pragma omp for schedule(runtime)
            for (int k = 0; k < local_size; ++k) {
                if (list->failed)
                    continue;
                int v = w->frontier[rank + k * processes];
                list->scanned += g->offsets[v + 1] - g->offsets[v];
                for (int j = g->offsets[v]; j < g->offsets[v + 1]; ++j) {
                    int u = g->edges[j];
                    if (dist[u] == -1 && !append(list, u))
                        break;
                }
            }
        }
        for (int t = 0; t < w->max_threads; ++t) {
            List *list = &w->lists[t];
            if (list->failed)
                fail("Cannot grow a thread candidate list.");
            w->scanned += list->scanned;
            for (size_t j = 0; j < list->size; ++j)
                w->local[list->data[j]] = 1;
        }
        MPI_Allreduce(w->local, w->global, g->n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int next_size = 0;
        for (int u = 0; u < g->n; ++u) {
            if (w->global[u] && dist[u] == -1) {
                dist[u] = level + 1;
                w->next[next_size++] = u;
            }
        }
        int *temporary = w->frontier;
        w->frontier = w->next;
        w->next = temporary;
        size = next_size;
    }
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
        fail("MPI_THREAD_FUNNELED is not available.");
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    if (argc < 3 || argc > 6) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s graph.txt source [undirected|directed] [dynamic|static] [repeats]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int source = parse_number(argv[2], 0, INT_MAX - 1);
    const char *mode = argc > 3 ? argv[3] : "undirected";
    const char *schedule = argc > 4 ? argv[4] : "dynamic";
    int repeats = argc > 5 ? parse_number(argv[5], 1, 1000) : 5;
    if (strcmp(mode, "directed") && strcmp(mode, "undirected"))
        fail("Graph mode must be directed or undirected.");
    if (strcmp(schedule, "static") && strcmp(schedule, "dynamic"))
        fail("Schedule must be static or dynamic.");
    omp_set_dynamic(0);
    omp_set_schedule(!strcmp(schedule, "static") ? omp_sched_static : omp_sched_dynamic,
                     !strcmp(schedule, "static") ? 0 : 1);
    Graph g = {0};
    if (rank == 0)
        g = read_graph(argv[1], !strcmp(mode, "directed"));
    MPI_Bcast(&g.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.arcs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (source >= g.n)
        fail("Source vertex is outside the graph.");
    if (rank != 0) {
        g.offsets = allocate((size_t)g.n + 1, sizeof(int));
        g.edges = allocate((size_t)g.arcs, sizeof(int));
    }
    MPI_Bcast(g.offsets, g.n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.edges, g.arcs, MPI_INT, 0, MPI_COMM_WORLD);

    int *reference = allocate((size_t)g.n, sizeof(int));
    int *dist = allocate((size_t)g.n, sizeof(int));
    int *queue = allocate((size_t)g.n, sizeof(int));
    Workspace w = {0};
    w.max_threads = omp_get_max_threads();
    w.frontier = allocate((size_t)g.n, sizeof(int));
    w.next = allocate((size_t)g.n, sizeof(int));
    w.local = allocate((size_t)g.n, sizeof(int));
    w.global = allocate((size_t)g.n, sizeof(int));
    w.sizes = allocate((size_t)g.n, sizeof(int));
    w.lists = allocate((size_t)w.max_threads, sizeof(List));
    double seq_total = 0;
    if (rank == 0) {
        sequential_bfs(&g, source, reference, queue);
        for (int run = 0; run < repeats; ++run) {
            double start = MPI_Wtime();
            sequential_bfs(&g, source, reference, queue);
            seq_total += MPI_Wtime() - start;
        }
    }
    MPI_Bcast(reference, g.n, MPI_INT, 0, MPI_COMM_WORLD);
    hybrid_bfs(&g, source, dist, &w, rank, processes);
    double par_total = 0;
    int max_mismatches = 0;
    for (int run = 0; run < repeats; ++run) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        hybrid_bfs(&g, source, dist, &w, rank, processes);
        double elapsed = MPI_Wtime() - start, slowest = 0;
        MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        par_total += slowest;
        int mismatches = 0, global_mismatches;
        for (int i = 0; i < g.n; ++i)
            mismatches += dist[i] != reference[i];
        MPI_Allreduce(&mismatches, &global_mismatches, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_mismatches > max_mismatches)
            max_mismatches = global_mismatches;
    }
    long long *counts = allocate((size_t)processes * 2, sizeof(long long));
    long long local_counts[2] = {w.processed, w.scanned};
    MPI_Gather(local_counts, 2, MPI_LONG_LONG_INT, counts, 2,
               MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    int min_threads, max_threads;
    MPI_Reduce(&w.actual_threads, &min_threads, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&w.actual_threads, &max_threads, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int unreachable = 0, max_frontier = 0;
        for (int i = 0; i < g.n; ++i)
            unreachable += reference[i] == -1;
        for (int i = 0; i < w.levels; ++i)
            if (w.sizes[i] > max_frontier)
                max_frontier = w.sizes[i];
        double seq = seq_total / repeats, par = par_total / repeats;
        printf("N=%d M=%d arcs=%d source=%d mode=%s\n", g.n, g.m, g.arcs, source, mode);
        printf("P=%d threads_min=%d threads_max=%d schedule=%s repeats=%d\n",
               processes, min_threads, max_threads, schedule, repeats);
        printf("levels=%d unreachable=%d mismatches=%d max_frontier=%d\n",
               w.levels, unreachable, max_mismatches, max_frontier);
        printf("T_seq=%.9f T_par=%.9f speedup=%.6f efficiency=%.6f\n",
               seq, par, seq / par, seq / par / processes / max_threads);
        for (int i = 0; i < w.levels; ++i)
            printf("FRONTIER level=%d size=%d\n", i, w.sizes[i]);
        for (int r = 0; r < processes; ++r)
            printf("WORK rank=%d vertices=%lld arcs=%lld\n", r, counts[2*r], counts[2*r+1]);
        if (g.n <= 64)
            for (int i = 0; i < g.n; ++i)
                printf("DIST vertex=%d seq=%d par=%d\n", i, reference[i], dist[i]);
        printf("Check=%s\n", max_mismatches == 0 ? "OK" : "FAIL");
    }
    for (int t = 0; t < w.max_threads; ++t)
        free(w.lists[t].data);
    free(w.lists);
    free(w.frontier);
    free(w.next);
    free(w.local);
    free(w.global);
    free(w.sizes);
    free(counts);
    free(queue);
    free(dist);
    free(reference);
    free(g.offsets);
    free(g.edges);
    MPI_Finalize();
    return max_mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
