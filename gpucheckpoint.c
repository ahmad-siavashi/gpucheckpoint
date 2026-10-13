/*
 * gpucheckpoint.c - in-process CUDA checkpoint/restore API.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Ahmad Siavashi
 */

#define _GNU_SOURCE

#include "gpucheckpoint.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_ILLEGAL_STATE 400
#define CUDA_TOOLKIT_VERSION 13000
#define MIN_DRIVER_VERSION 5500

typedef int cu_result_t;

typedef struct {
    char bytes[16];
} cu_uuid_t;

typedef struct {
    unsigned int timeout_ms;
    unsigned int reserved0;
    unsigned long long reserved1[7];
} lock_args_t;

typedef struct {
    unsigned long long reserved[8];
} checkpoint_args_t;

typedef struct {
    cu_uuid_t old_uuid;
    cu_uuid_t new_uuid;
} gpu_pair_t;

typedef struct {
    gpu_pair_t *gpu_pairs;
    unsigned int gpu_pairs_count;
    char reserved[44];
    unsigned long long reserved1;
} restore_args_t;

typedef struct {
    unsigned long long reserved[8];
} unlock_args_t;

typedef cu_result_t (*fn_get_proc_address)(const char *, void **, int,
                                           unsigned long long);
typedef cu_result_t (*fn_init)(unsigned int);
typedef cu_result_t (*fn_driver_version)(int *);
typedef cu_result_t (*fn_error_string)(cu_result_t, const char **);
typedef cu_result_t (*fn_get_state)(int, int *);
typedef cu_result_t (*fn_get_restore_tid)(int, int *);
typedef cu_result_t (*fn_lock)(int, lock_args_t *);
typedef cu_result_t (*fn_checkpoint)(int, checkpoint_args_t *);
typedef cu_result_t (*fn_restore)(int, restore_args_t *);
typedef cu_result_t (*fn_unlock)(int, unlock_args_t *);

static int ready = 0;
static int driver_inited = 0;
static void *driver_handle;

static fn_get_proc_address p_cu_get_proc_address;
static fn_init p_cu_init;
static fn_driver_version p_cu_driver_version;
static fn_error_string p_cu_get_error_string;
static fn_get_state p_get_state;
static fn_get_restore_tid p_get_restore_tid;
static fn_lock p_lock;
static fn_checkpoint p_checkpoint;
static fn_restore p_restore;
static fn_unlock p_unlock;

static void *resolve_symbol(const char *name)
{
    void *sym = NULL;

    if (p_cu_get_proc_address != NULL &&
        p_cu_get_proc_address(name, &sym, CUDA_TOOLKIT_VERSION, 0) == 0 &&
        sym != NULL) {
        return sym;
    }
    return dlsym(driver_handle, name);
}

int gc_init(void)
{
    void *handle;

    if (ready) {
        return GC_OK;
    }

    handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (handle == NULL) {
        handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (handle == NULL) {
        return GC_ERR_NOT_READY;
    }
    driver_handle = handle;
    p_cu_get_proc_address = (fn_get_proc_address)dlsym(handle, "cuGetProcAddress");

    p_cu_init = (fn_init)resolve_symbol("cuInit");
    p_cu_driver_version = (fn_driver_version)resolve_symbol("cuDriverGetVersion");
    p_cu_get_error_string = (fn_error_string)resolve_symbol("cuGetErrorString");
    p_get_state = (fn_get_state)resolve_symbol("cuCheckpointProcessGetState");
    p_get_restore_tid = (fn_get_restore_tid)resolve_symbol("cuCheckpointProcessGetRestoreThreadId");
    p_lock = (fn_lock)resolve_symbol("cuCheckpointProcessLock");
    p_checkpoint = (fn_checkpoint)resolve_symbol("cuCheckpointProcessCheckpoint");
    p_restore = (fn_restore)resolve_symbol("cuCheckpointProcessRestore");
    p_unlock = (fn_unlock)resolve_symbol("cuCheckpointProcessUnlock");

    if (p_cu_init == NULL || p_get_state == NULL || p_get_restore_tid == NULL ||
        p_lock == NULL || p_checkpoint == NULL || p_restore == NULL ||
        p_unlock == NULL) {
        return GC_ERR_NOT_READY;
    }

    if (p_cu_driver_version != NULL) {
        int version = 0;

        if (p_cu_driver_version(&version) == 0 && version > 0 &&
            version < MIN_DRIVER_VERSION) {
            return GC_ERR_NOT_READY;
        }
    }

    ready = 1;
    return GC_OK;
}

static int ensure_driver_init(void)
{
    cu_result_t rc;

    if (driver_inited) {
        return GC_OK;
    }
    rc = p_cu_init(0);
    if (rc != 0) {
        return rc;
    }
    driver_inited = 1;
    return GC_OK;
}

int gc_get_state(int pid, int *state)
{
    cu_result_t rc;

    if (!ready) {
        return GC_ERR_NOT_READY;
    }
    if (state == NULL) {
        return GC_ERR_BAD_ARGS;
    }
    rc = p_get_state(pid, state);
    if (rc != 0 && rc == CUDA_ERROR_NOT_INITIALIZED && !driver_inited) {
        if (ensure_driver_init() != GC_OK) {
            return GC_ERR_NOT_READY;
        }
        rc = p_get_state(pid, state);
    }
    return rc;
}

int gc_get_restore_tid(int pid, int *tid)
{
    cu_result_t rc;

    if (!ready) {
        return GC_ERR_NOT_READY;
    }
    if (tid == NULL) {
        return GC_ERR_BAD_ARGS;
    }
    rc = p_get_restore_tid(pid, tid);
    if (rc != 0 && rc == CUDA_ERROR_NOT_INITIALIZED && !driver_inited) {
        if (ensure_driver_init() != GC_OK) {
            return GC_ERR_NOT_READY;
        }
        rc = p_get_restore_tid(pid, tid);
    }
    return rc;
}

int gc_lock(int pid, unsigned int timeout_ms)
{
    lock_args_t args = {0};
    cu_result_t rc;

    if (!ready) {
        return GC_ERR_NOT_READY;
    }
    if (ensure_driver_init() != GC_OK) {
        return GC_ERR_NOT_READY;
    }
    args.timeout_ms = timeout_ms;
    rc = p_lock(pid, &args);
    return rc;
}

int gc_checkpoint(int pid)
{
    checkpoint_args_t args = {0};
    cu_result_t rc;

    if (!ready) {
        return GC_ERR_NOT_READY;
    }
    if (ensure_driver_init() != GC_OK) {
        return GC_ERR_NOT_READY;
    }
    rc = p_checkpoint(pid, &args);
    return rc;
}

int gc_restore(int pid, const gc_gpu_pair_t *pairs, unsigned int pair_count)
{
    restore_args_t args = {0};
    cu_result_t rc;

    if (!ready) {
        return GC_ERR_NOT_READY;
    }
    if (pair_count != 0 && pairs == NULL) {
        return GC_ERR_BAD_ARGS;
    }
    if (ensure_driver_init() != GC_OK) {
        return GC_ERR_NOT_READY;
    }
    args.gpu_pairs = (gpu_pair_t *)pairs;
    args.gpu_pairs_count = pair_count;
    rc = p_restore(pid, &args);
    return rc;
}

int gc_unlock(int pid)
{
    unlock_args_t args = {0};
    cu_result_t rc;

    if (!ready) {
        return GC_ERR_NOT_READY;
    }
    if (ensure_driver_init() != GC_OK) {
        return GC_ERR_NOT_READY;
    }
    rc = p_unlock(pid, &args);
    return rc;
}

int gc_toggle(int pid, const gc_gpu_pair_t *pairs, unsigned int pair_count)
{
    int state = 0;
    int rc;

    if (!ready) {
        return GC_ERR_NOT_READY;
    }
    rc = gc_get_state(pid, &state);
    if (rc != GC_OK) {
        return rc;
    }
    if (state == GC_STATE_RUNNING) {
        rc = gc_lock(pid, 0);
        if (rc != GC_OK) {
            return rc;
        }
        return gc_checkpoint(pid);
    }
    if (state == GC_STATE_CHECKPOINTED) {
        rc = gc_restore(pid, pairs, pair_count);
        if (rc != GC_OK) {
            return rc;
        }
        return gc_unlock(pid);
    }
    return CUDA_ERROR_ILLEGAL_STATE;
}

static int is_hex(char c)
{
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static int parse_uuid(const char *text, char *uuid)
{
    char digits[33];
    int i, n = 0;

    if (text == NULL || strncasecmp(text, "GPU-", 4) != 0) {
        return -1;
    }
    text += 4;
    for (i = 0; text[i] != '\0'; i++) {
        if (text[i] == '-') {
            continue;
        }
        if (!is_hex(text[i]) || n >= 32) {
            return -1;
        }
        digits[n++] = text[i];
    }
    if (n != 32) {
        return -1;
    }
    digits[32] = '\0';
    for (i = 0; i < 16; i++) {
        unsigned int byte = 0;

        sscanf(digits + i * 2, "%2x", &byte);
        uuid[i] = (char)byte;
    }
    return 0;
}

int gc_parse_device_map(const char *map, gc_gpu_pair_t *pairs,
                        unsigned int pair_capacity, unsigned int *pair_count)
{
    char *copy;
    char *save = NULL;
    char *token;
    unsigned int count = 0;

    if (map == NULL || pairs == NULL || pair_count == NULL || pair_capacity == 0) {
        return GC_ERR_BAD_ARGS;
    }

    copy = strdup(map);
    if (copy == NULL) {
        return GC_ERR_BAD_ARGS;
    }

    for (token = strtok_r(copy, ",", &save); token != NULL;
         token = strtok_r(NULL, ",", &save)) {
        char *eq = strchr(token, '=');
        char *old_text;
        char *new_text;

        if (eq == NULL || eq == token || eq[1] == '\0') {
            goto fail;
        }
        if (count >= pair_capacity) {
            goto fail;
        }
        *eq = '\0';
        old_text = token;
        new_text = eq + 1;

        if (parse_uuid(old_text, pairs[count].old_uuid) != 0) {
            goto fail;
        }
        if (parse_uuid(new_text, pairs[count].new_uuid) != 0) {
            goto fail;
        }
        count++;
    }

    if (count == 0) {
        goto fail;
    }

    free(copy);
    *pair_count = count;
    return GC_OK;

fail:
    free(copy);
    return GC_ERR_BAD_ARGS;
}

const char *gc_state_name(int state)
{
    switch (state) {
    case GC_STATE_RUNNING:
        return "running";
    case GC_STATE_LOCKED:
        return "locked";
    case GC_STATE_CHECKPOINTED:
        return "checkpointed";
    case GC_STATE_FAILED:
        return "failed";
    case GC_STATE_CHECKPOINTING:
        return "checkpointing";
    case GC_STATE_RESTORING:
        return "restoring";
    default:
        return "unknown";
    }
}

const char *gc_error_string(int rc)
{
    static char buffer[64];
    const char *str = NULL;

    switch (rc) {
    case GC_OK:
        return "no error";
    case GC_ERR_NOT_READY:
        return "driver not loaded or too old";
    case GC_ERR_BAD_ARGS:
        return "invalid arguments";
    default:
        break;
    }

    if (rc > 0 && p_cu_get_error_string != NULL &&
        p_cu_get_error_string(rc, &str) == 0 && str != NULL) {
        return str;
    }
    snprintf(buffer, sizeof(buffer), "driver error %d", rc);
    return buffer;
}
