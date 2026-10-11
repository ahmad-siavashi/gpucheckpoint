/*
 * lib-test.c - test the in-process library against a running CUDA process.
 *
 * SPDX-License-Identifier: MIT
 *
 * Usage: lib-test <pid-of-running-cuda-process>
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

    if (argc != 2) {
        fprintf(stderr, "usage: %s <pid>\n", argv[0]);
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
