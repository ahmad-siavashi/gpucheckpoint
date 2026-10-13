/*
 * lib-test.c - test the in-process library against a running CUDA process.
 *
 * SPDX-License-Identifier: MIT
 *
 * Usage: lib-test <pid-of-running-cuda-process> [gpu-uuid]
 * With a GPU UUID it also checks that a driver error keeps its own code.
 * Runs the full state machine in-process (no CLI) and checks that the
 * process returns to a running state at the end.
 */

#include "gpucheckpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect(int pid, const char *name, int rc, int want_state)
{
    int state = -1;

    if (rc != GC_OK) {
        printf("FAIL  %s: rc=%d (%s)\n", name, rc, gc_error_string(rc));
        failures++;
        return;
    }
    if (want_state >= 0) {
        if (gc_get_state(pid, &state) != GC_OK || state != want_state) {
            printf("FAIL  %s: state=%s\n", name, gc_state_name(state));
            failures++;
            return;
        }
    }
    printf("PASS  %s\n", name);
}

int main(int argc, char **argv)
{
    int pid;
    int state;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <pid> [gpu-uuid]\n", argv[0]);
        return 2;
    }
    pid = atoi(argv[1]);

    if (gc_init() != GC_OK) {
        printf("FAIL  gc_init: %s\n", gc_error_string(GC_ERR_NOT_READY));
        return 1;
    }
    printf("PASS  gc_init\n");

    if (gc_get_state(pid, &state) != GC_OK || state != GC_STATE_RUNNING) {
        printf("FAIL  precondition: process is %s\n", gc_state_name(state));
        return 1;
    }

    expect(pid, "lock", gc_lock(pid, 0), GC_STATE_LOCKED);
    expect(pid, "checkpoint", gc_checkpoint(pid), GC_STATE_CHECKPOINTED);
    expect(pid, "restore", gc_restore(pid, NULL, 0), GC_STATE_LOCKED);
    expect(pid, "unlock", gc_unlock(pid), GC_STATE_RUNNING);
    if (argc == 3) {
        gc_gpu_pair_t same;
        unsigned int n = 0;
        char map[160];
        int rc;

        /* The driver rejects a same-GPU remap; its code must not be
         * mistaken for GC_ERR_NOT_READY or GC_ERR_BAD_ARGS. */
        snprintf(map, sizeof(map), "%s=%s", argv[2], argv[2]);
        gc_lock(pid, 0);
        gc_checkpoint(pid);
        if (gc_parse_device_map(map, &same, 1, &n) != GC_OK) {
            printf("FAIL  parse same-GPU map\n");
            failures++;
        } else {
            rc = gc_restore(pid, &same, n);
            if (rc > 0) {
                printf("PASS  driver rejection keeps its own code (%d: %s)\n",
                       rc, gc_error_string(rc));
            } else {
                printf("FAIL  driver rejection returned rc=%d (%s)\n",
                       rc, gc_error_string(rc));
                failures++;
            }
        }
        gc_restore(pid, NULL, 0);
        gc_unlock(pid);
    }

    expect(pid, "toggle off", gc_toggle(pid, NULL, 0), GC_STATE_CHECKPOINTED);
    expect(pid, "toggle on", gc_toggle(pid, NULL, 0), GC_STATE_RUNNING);

    {
        gc_gpu_pair_t pairs[4];
        unsigned int count = 0;

        if (gc_parse_device_map("GPU-01=GPU-02", pairs, 4, &count) == GC_OK) {
            printf("FAIL  parse_device_map rejects short uuids\n");
            failures++;
        } else {
            printf("PASS  parse_device_map rejects short uuids\n");
        }
        if (gc_parse_device_map("garbage", pairs, 4, &count) == GC_OK) {
            printf("FAIL  parse_device_map rejects garbage\n");
            failures++;
        } else {
            printf("PASS  parse_device_map rejects garbage\n");
        }
    }

    if (gc_get_state(pid, &state) != GC_OK || state != GC_STATE_RUNNING) {
        printf("FAIL  process left in state %s\n", gc_state_name(state));
        return 1;
    }
    printf("PASS  process left running\n");

    if (failures != 0) {
        printf("lib-test: %d failures\n", failures);
        return 1;
    }
    printf("lib-test: all passed\n");
    return 0;
}
