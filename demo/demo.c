/*
 * demo.c - embed the in-process checkpoint library in a program.
 *
 * SPDX-License-Identifier: MIT
 *
 * Launches a CUDA program (e.g. a build of tests/counter.c) as a child, then checkpoints and restores it
 * directly (no CLI, no process spawn per operation) and prints the time
 * each operation took.
 */

#define _GNU_SOURCE

#include "gpucheckpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static unsigned long long now_us(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000ULL +
           (unsigned long long)ts.tv_nsec / 1000ULL;
}

static void timed(const char *name, int rc)
{
    printf("%-10s rc=%d (%s)\n", name, rc, gc_error_string(rc));
}

int main(int argc, char **argv)
{
    const char *counter;
    pid_t child;
    unsigned long long t0, t1;
    int rc, state, tid;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <cuda-program>\n", argv[0]);
        return 2;
    }
    counter = argv[1];

    rc = gc_init();
    if (rc != GC_OK) {
        fprintf(stderr, "gc_init failed: %s\n", gc_error_string(rc));
        return 1;
    }

    child = fork();
    if (child == 0) {
        execl(counter, counter, (char *)NULL);
        _exit(127);
    }

    for (;;) {
        int status;

        if (waitpid(child, &status, WNOHANG) == child) {
            fprintf(stderr, "counter exited early\n");
            return 1;
        }
        rc = gc_get_state(child, &state);
        if (rc == GC_OK && state == GC_STATE_RUNNING) {
            break;
        }
        usleep(100000);
    }
    printf("counter pid=%d is running\n", (int)child);

    gc_get_state(child, &state);
    t0 = now_us();
    for (int i = 0; i < 100; i++) {
        gc_get_state(child, &state);
    }
    t1 = now_us();
    printf("gc_get_state x100 in-process: %llu us total (%llu us mean)\n",
           t1 - t0, (t1 - t0) / 100);

    rc = gc_get_restore_tid(child, &tid);
    printf("restore tid: %d (rc=%d)\n", tid, rc);

    timed("lock", gc_lock(child, 0));
    timed("checkpoint", gc_checkpoint(child));
    rc = gc_get_state(child, &state);
    printf("state after checkpoint: %s\n", gc_state_name(state));
    sleep(2);

    timed("restore", gc_restore(child, NULL, 0));
    timed("unlock", gc_unlock(child));
    rc = gc_get_state(child, &state);
    printf("state after restore+unlock: %s\n", gc_state_name(state));

    kill(child, 9);
    waitpid(child, NULL, 0);
    return 0;
}
