/*
 * gpucheckpoint.h - in-process CUDA checkpoint/restore API.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Ahmad Siavashi
 *
 * Embeddable equivalent of the cuda-checkpoint CLI. All work is performed
 * by the NVIDIA display driver through the public cuCheckpointProcess*
 * driver API. Call gc_init() once, then call the operations directly -
 * no process is spawned.
 *
 * Every function returns GC_OK (0) on success. On failure it returns
 * GC_ERR_NOT_READY or GC_ERR_BAD_ARGS (negative), or the CUDA driver error
 * code as is (positive). Pass any of them to gc_error_string() for the
 * message.
 *
 * Not thread-safe: serialise calls, including gc_init() and
 * gc_error_string() (it returns a static buffer for unknown codes).
 */

#ifndef GPUCHECKPOINT_H
#define GPUCHECKPOINT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GC_OK 0
#define GC_ERR_NOT_READY -1
#define GC_ERR_BAD_ARGS -2

enum {
    GC_STATE_RUNNING = 0,
    GC_STATE_LOCKED = 1,
    GC_STATE_CHECKPOINTED = 2,
    GC_STATE_FAILED = 3,
    GC_STATE_CHECKPOINTING = 4,
    GC_STATE_RESTORING = 5,
};

/* A (old GPU, new GPU) pair for remapping during gc_restore(). The uuid
 * fields hold the raw 16-byte GPU UUID. Use gc_parse_device_map() to fill
 * pairs from the "GPU-<32 hex>=GPU-<32 hex>" command-line format (dashes
 * as printed by nvidia-smi are accepted). */
typedef struct {
    char old_uuid[16];
    char new_uuid[16];
} gc_gpu_pair_t;

/* Load libcuda.so.1 and resolve the checkpoint API. Safe to call more
 * than once. Must succeed before any other call. */
int gc_init(void);

/* Return the checkpoint state (GC_STATE_*) of process pid. */
int gc_get_state(int pid, int *state);

/* Return the CUDA restore thread ID of process pid. */
int gc_get_restore_tid(int pid, int *tid);

/* Lock process pid. timeout_ms is 0 for no timeout. On timeout the
 * process is left in its previous state. */
int gc_lock(int pid, unsigned int timeout_ms);

/* Checkpoint process pid (must be locked). GPU memory is moved to host
 * memory and all GPU resources of the process are released. */
int gc_checkpoint(int pid);

/* Restore process pid (must be checkpointed). If pair_count is nonzero,
 * pairs remaps checkpointed GPUs onto other physical GPUs and must name
 * every checkpointed GPU. */
int gc_restore(int pid, const gc_gpu_pair_t *pairs, unsigned int pair_count);

/* Unlock process pid (must be locked). */
int gc_unlock(int pid);

/* Switch pid between running and checkpointed in one call. */
int gc_toggle(int pid, const gc_gpu_pair_t *pairs, unsigned int pair_count);

/* Parse a "GPU-<32 hex>=GPU-<32 hex>,..." device map into pairs.
 * pair_capacity is the size of pairs. On success *pair_count is set. */
int gc_parse_device_map(const char *map, gc_gpu_pair_t *pairs,
                        unsigned int pair_capacity, unsigned int *pair_count);

/* "running", "locked", "checkpointed", "failed", "checkpointing",
 * "restoring", or "unknown". */
const char *gc_state_name(int state);

/* Error message for a GC_* or driver return code (GC_OK gives "no error"). */
const char *gc_error_string(int rc);

#ifdef __cplusplus
}
#endif

#endif /* GPUCHECKPOINT_H */
