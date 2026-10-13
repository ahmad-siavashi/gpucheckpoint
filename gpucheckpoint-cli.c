/*
 * gpucheckpoint-cli.c - open CUDA checkpoint/restore CLI.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Ahmad Siavashi
 *
 * Independent implementation of a CUDA checkpoint/restore command-line
 * interface. All GPU-side work is performed by the NVIDIA display driver
 * through the public CUDA driver API (cuCheckpointProcess*), resolved at
 * runtime from libcuda.so.1. No NVIDIA code is included, linked, or
 * redistributed by this program, and no text from NVIDIA tools or
 * documentation is reproduced.
 */

#define _GNU_SOURCE

#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define MIN_DRIVER_VERSION 5500
#define CUDA_ERROR_NOT_INITIALIZED 3
#define TOOL_VERSION "1.1.0"

#define EXIT_USAGE 2

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

#define CUDA_TOOLKIT_VERSION 13000

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
typedef cu_result_t (*fn_device_get_count)(int *);
typedef cu_result_t (*fn_device_get_uuid)(cu_uuid_t *, int);

static int driver_inited = 0;
static int trace_on = 0;
static void *driver_handle;
static fn_get_proc_address p_cu_get_proc_address;

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

static fn_init p_cu_init;
static fn_driver_version p_cu_driver_version;
static fn_error_string p_cu_get_error_string;
static fn_get_state p_get_state;
static fn_get_restore_tid p_get_restore_tid;
static fn_lock p_lock;
static fn_checkpoint p_checkpoint;
static fn_restore p_restore;
static fn_unlock p_unlock;
static fn_device_get_count p_cu_device_get_count;
static fn_device_get_uuid p_cu_device_get_uuid;

enum {
    STATE_RUNNING = 0,
    STATE_LOCKED = 1,
    STATE_CHECKPOINTED = 2,
    STATE_FAILED = 3,
    STATE_CHECKPOINTING = 4,
    STATE_RESTORING = 5,
};

enum {
    CMD_NONE = 0,
    CMD_STATE,
    CMD_SAVE,
    CMD_RESUME,
    CMD_LOCK,
    CMD_CHECKPOINT,
    CMD_RESTORE,
    CMD_UNLOCK,
    CMD_TOGGLE,
    CMD_RESTORE_TID,
    CMD_GPUS,
    CMD_SCAN,
};

struct command {
    int id;
    const char *name;
    const char *summary;
};

static const struct command commands[] = {
    { CMD_STATE,       "state",       "Show the checkpoint state of the process" },
    { CMD_SAVE,        "save",        "Lock and checkpoint the process in one call" },
    { CMD_RESUME,      "resume",      "Restore and unlock the process in one call" },
    { CMD_LOCK,        "lock",        "Lock the process" },
    { CMD_CHECKPOINT,  "checkpoint",  "Move GPU memory to host memory and release GPU resources" },
    { CMD_RESTORE,     "restore",     "Put the GPU memory back, optionally onto other GPUs" },
    { CMD_UNLOCK,      "unlock",      "Unlock the process" },
    { CMD_TOGGLE,      "toggle",      "Alias for save or resume, depending on the current state" },
    { CMD_RESTORE_TID, "restore-tid", "Show the CUDA restore thread ID of the process" },
    { CMD_GPUS,        "gpus",        "List the GPUs with their index and UUID" },
    { CMD_SCAN,        "scan",        "Show the checkpoint state of every CUDA process" },
};

static const char *state_name(int state)
{
    switch (state) {
    case STATE_RUNNING:
        return "running";
    case STATE_LOCKED:
        return "locked";
    case STATE_CHECKPOINTED:
        return "checkpointed";
    case STATE_FAILED:
        return "failed";
    case STATE_CHECKPOINTING:
        return "checkpointing";
    case STATE_RESTORING:
        return "restoring";
    default:
        return "unknown";
    }
}

static const char *error_string(cu_result_t rc)
{
    const char *str = NULL;

    if (p_cu_get_error_string && p_cu_get_error_string(rc, &str) == 0 && str != NULL) {
        return str;
    }
    return "unknown driver error";
}

static unsigned long long now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000ULL +
           (unsigned long long)ts.tv_nsec / 1000000ULL;
}

static void trace_call(const char *name, const char *detail, cu_result_t rc,
                       unsigned long long t0)
{
    if (trace_on) {
        fprintf(stderr, "[trace] %s(%s) -> %s (%llu ms)\n", name, detail,
                error_string(rc), now_ms() - t0);
    }
}

static const char *driver_version(void)
{
    static char version[32];
    int fd = open("/proc/driver/nvidia/version", O_RDONLY);

    if (fd < 0) {
        return "unknown";
    }
    {
        char buf[512] = {0};
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n > 0) {
            char *p = buf;
            while (*p) {
                if (isdigit((unsigned char)*p)) {
                    int major = 0, minor = 0, patch = 0;
                    if (sscanf(p, "%d.%d.%d", &major, &minor, &patch) == 3) {
                        snprintf(version, sizeof(version), "%d.%d.%02d", major, minor, patch);
                        return version;
                    }
                }
                p++;
            }
        }
    }
    return "unknown";
}

static void help(FILE *out)
{
    size_t i;

    fprintf(out,
            "gpucheckpoint - checkpoint and restore the GPU state of a process\n"
            "\n"
            "Usage:\n"
            "  gpucheckpoint <command> [options]\n"
            "\n"
            "Commands:\n");
    for (i = 0; i < sizeof(commands) / sizeof(commands[0]); i++) {
        fprintf(out, "  %-13s %s\n", commands[i].name, commands[i].summary);
    }
    fprintf(out,
            "\n"
            "Options:\n"
            "  -p, --pid <pid>         Process ID (required by the process commands)\n"
            "  -t, --timeout <ms>      Lock timeout in milliseconds (lock only)\n"
            "  -d, --device-map <map>  Old=new GPU pairs, index or UUID, comma separated\n"
            "                          (restore, resume, toggle). The target GPU must be\n"
            "                          the same chip type with enough memory.\n"
            "      --json              Machine-readable output (state, restore-tid, scan)\n"
            "      --trace             Print each driver call with result and duration\n"
            "  -h, --help              Show help for this command\n"
            "      --version           Show version information\n"
            "\n"
            "State machine:\n"
            "  running -> lock -> locked -> checkpoint -> checkpointed\n"
            "  checkpointed -> restore -> locked -> unlock -> running\n"
            "\n"
            "  save is lock+checkpoint in one call, resume is restore+unlock.\n"
            "  On partial failure they print the resulting state and the next\n"
            "  command to run.\n"
            "\n"
            "Examples:\n"
            "  gpucheckpoint state -p 1234\n"
            "  gpucheckpoint save -p 1234\n"
            "  gpucheckpoint resume -p 1234\n"
            "  gpucheckpoint scan\n");
}

static void cmd_help(FILE *out, int cmd)
{
    size_t i;

    for (i = 0; i < sizeof(commands) / sizeof(commands[0]); i++) {
        if (commands[i].id == cmd) {
            break;
        }
    }
    fprintf(out,
            "Usage:\n"
            "  gpucheckpoint %s [options]\n"
            "\n"
            "%s\n"
            "\n"
            "Options:\n",
            commands[i].name, commands[i].summary);
    if (cmd == CMD_GPUS) {
        fprintf(out,
                "      --trace        Print each driver call with result and duration\n"
                "  -h, --help         Show this help\n");
        return;
    }
    if (cmd == CMD_SCAN) {
        fprintf(out,
                "      --json         Machine-readable output\n"
                "      --trace        Print each driver call with result and duration\n"
                "  -h, --help         Show this help\n");
        return;
    }
    fprintf(out, "  -p, --pid <pid>    Process ID (required)\n");
    if (cmd == CMD_LOCK) {
        fprintf(out,
                "  -t, --timeout <ms> Lock timeout in milliseconds\n"
                "                     (default: no timeout)\n");
    }
    if (cmd == CMD_RESTORE || cmd == CMD_TOGGLE || cmd == CMD_RESUME) {
        fprintf(out,
                "  -d, --device-map <map>\n"
                "      Old=new GPU pairs, index or UUID, comma separated. Must name\n"
                "      every checkpointed GPU.\n");
    }
    if (cmd == CMD_STATE || cmd == CMD_RESTORE_TID) {
        fprintf(out, "      --json         Machine-readable output\n");
    }
    fprintf(out,
            "      --trace        Print each driver call with result and duration\n"
            "  -h, --help         Show this help\n");
}

static int usage_error(const char *message)
{
    if (message != NULL && message[0] != '\0') {
        fprintf(stderr, "%s\n\n", message);
    }
    help(stderr);
    return EXIT_USAGE;
}

static void print_version(void)
{
    printf("gpucheckpoint %s\n", TOOL_VERSION);
    printf("driver %s\n", driver_version());
}

static int parse_pid(const char *text, int *pid)
{
    char *end = NULL;
    long value = strtol(text, &end, 10);

    if (end == text || *end != '\0' || value < 0 || value > 0x7fffffff) {
        return -1;
    }
    *pid = (int)value;
    return 0;
}

static int is_hex(char c)
{
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static int parse_uuid(const char *text, cu_uuid_t *uuid)
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
        uuid->bytes[i] = (char)byte;
    }
    return 0;
}

static int resolve_uuid(const char *text, cu_uuid_t *uuid)
{
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);

    if (end != text && *end == '\0' && value < 64) {
        int count = 0;

        if (p_cu_device_get_count != NULL && p_cu_device_get_uuid != NULL &&
            p_cu_device_get_count(&count) == 0 && (int)value < count) {
            if (p_cu_device_get_uuid(uuid, (int)value) == 0) {
                return 0;
            }
        }
        return -1;
    }
    return parse_uuid(text, uuid);
}

static void format_uuid(const cu_uuid_t *uuid, char *out)
{
    int i;

    for (i = 0; i < 16; i++) {
        sprintf(out + i * 2, "%02x", (unsigned char)uuid->bytes[i]);
    }
    out[32] = '\0';
}

static int parse_device_map(const char *map, gpu_pair_t **pairs_out, unsigned int *count_out)
{
    char *copy;
    char *save = NULL;
    char *token;
    gpu_pair_t *pairs;
    unsigned int count = 0;
    unsigned int capacity;

    copy = strdup(map);
    if (copy == NULL) {
        return -1;
    }

    capacity = 8;
    pairs = malloc(capacity * sizeof(pairs[0]));
    if (pairs == NULL) {
        free(copy);
        return -1;
    }

    for (token = strtok_r(copy, ",", &save); token != NULL; token = strtok_r(NULL, ",", &save)) {
        char *eq = strchr(token, '=');
        char *old_text;
        char *new_text;

        if (eq == NULL || eq == token || eq[1] == '\0') {
            goto fail;
        }
        *eq = '\0';
        old_text = token;
        new_text = eq + 1;

        if (resolve_uuid(old_text, &pairs[count].old_uuid) != 0) {
            goto fail;
        }
        if (resolve_uuid(new_text, &pairs[count].new_uuid) != 0) {
            goto fail;
        }
        count++;
        if (count >= capacity) {
            capacity *= 2;
            pairs = realloc(pairs, capacity * sizeof(pairs[0]));
            if (pairs == NULL) {
                goto fail;
            }
        }
    }

    if (count == 0) {
        goto fail;
    }

    free(copy);
    *pairs_out = pairs;
    *count_out = count;
    return 0;

fail:
    free(copy);
    free(pairs);
    return -1;
}

static void load_driver(void)
{
    void *handle;

    handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (handle == NULL) {
        handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (handle == NULL) {
        fprintf(stderr, "Error: could not load libcuda.so.1: %s\n", dlerror());
        exit(1);
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
    p_cu_device_get_count = (fn_device_get_count)resolve_symbol("cuDeviceGetCount");
    p_cu_device_get_uuid = (fn_device_get_uuid)resolve_symbol("cuDeviceGetUuid");

    if (p_cu_init == NULL || p_get_state == NULL || p_get_restore_tid == NULL ||
        p_lock == NULL || p_checkpoint == NULL || p_restore == NULL ||
        p_unlock == NULL) {
        fprintf(stderr,
                "Error: libcuda.so.1 does not export the checkpoint API "
                "(display driver 570 or higher required)\n");
        exit(1);
    }

    if (p_cu_driver_version != NULL) {
        int version = 0;

        if (p_cu_driver_version(&version) == 0 && version > 0 && version < MIN_DRIVER_VERSION) {
            fprintf(stderr,
                    "Error: display driver %d.%02d is too old, version 550 or higher is required\n",
                    version / 1000, version % 1000);
            exit(1);
        }
    }
}

static int ensure_driver_init(void)
{
    cu_result_t rc;
    unsigned long long t0 = now_ms();

    if (driver_inited) {
        return 0;
    }
    rc = p_cu_init(0);
    trace_call("cuInit", "", rc, t0);
    if (rc != 0) {
        fprintf(stderr, "Error: CUDA driver initialization failed\n");
        return 1;
    }
    driver_inited = 1;
    return 0;
}

static int do_lock(int pid, long timeout_ms)
{
    lock_args_t args = {0};
    cu_result_t rc;
    char detail[32];
    unsigned long long t0;

    snprintf(detail, sizeof(detail), "pid=%d, timeout=%ld", pid, timeout_ms);
    t0 = now_ms();
    args.timeout_ms = (unsigned int)timeout_ms;
    rc = p_lock(pid, &args);
    trace_call("cuCheckpointProcessLock", detail, rc, t0);
    if (rc != 0) {
        fprintf(stderr, "failed to lock process %d: %s\n", pid, error_string(rc));
    }
    return rc == 0 ? 0 : 1;
}

static int do_checkpoint(int pid)
{
    checkpoint_args_t args = {0};
    cu_result_t rc;
    char detail[32];
    unsigned long long t0;

    snprintf(detail, sizeof(detail), "pid=%d", pid);
    t0 = now_ms();
    rc = p_checkpoint(pid, &args);
    trace_call("cuCheckpointProcessCheckpoint", detail, rc, t0);
    if (rc != 0) {
        fprintf(stderr, "failed to checkpoint process %d: %s\n", pid, error_string(rc));
    }
    return rc == 0 ? 0 : 1;
}

static int do_restore(int pid, gpu_pair_t *pairs, unsigned int count)
{
    restore_args_t args = {0};
    cu_result_t rc;
    char detail[64];
    unsigned long long t0;

    args.gpu_pairs = pairs;
    args.gpu_pairs_count = count;
    snprintf(detail, sizeof(detail), "pid=%d, gpus=%u", pid, count);
    t0 = now_ms();
    rc = p_restore(pid, &args);
    trace_call("cuCheckpointProcessRestore", detail, rc, t0);
    if (rc != 0) {
        fprintf(stderr, "failed to restore process %d: %s\n", pid, error_string(rc));
    }
    return rc == 0 ? 0 : 1;
}

static int do_unlock(int pid)
{
    unlock_args_t args = {0};
    cu_result_t rc;
    char detail[32];
    unsigned long long t0;

    snprintf(detail, sizeof(detail), "pid=%d", pid);
    t0 = now_ms();
    rc = p_unlock(pid, &args);
    trace_call("cuCheckpointProcessUnlock", detail, rc, t0);
    if (rc != 0) {
        fprintf(stderr, "failed to unlock process %d: %s\n", pid, error_string(rc));
    }
    return rc == 0 ? 0 : 1;
}

static int query_state(int pid, int *state)
{
    cu_result_t rc = p_get_state(pid, state);

    if (rc != 0 && rc == CUDA_ERROR_NOT_INITIALIZED && !driver_inited &&
        ensure_driver_init() == 0) {
        rc = p_get_state(pid, state);
    }
    return rc == 0 ? 0 : 1;
}

static int state_guard(int pid, int want, const char *op)
{
    int state = 0;

    if (query_state(pid, &state) != 0) {
        return 0;
    }
    if (state == want) {
        return 0;
    }

    if (state == STATE_FAILED) {
        fprintf(stderr, "process %d is in the failed state; it cannot be recovered\n", pid);
    } else if (state == STATE_CHECKPOINTING || state == STATE_RESTORING) {
        fprintf(stderr, "process %d is %s; wait for the operation to finish\n",
                pid, state_name(state));
    } else if (state == STATE_RUNNING) {
        if (want == STATE_LOCKED && strcmp(op, "unlock") != 0) {
            fprintf(stderr, "process %d is running; run 'lock' first\n", pid);
        } else if (want == STATE_CHECKPOINTED) {
            fprintf(stderr, "process %d is running; run 'save' first\n", pid);
        } else {
            fprintf(stderr, "process %d is already running\n", pid);
        }
    } else if (state == STATE_LOCKED) {
        if (strcmp(op, "resume") == 0) {
            fprintf(stderr, "process %d is locked; run 'unlock' to finish\n", pid);
        } else if (strcmp(op, "save") == 0 || want == STATE_CHECKPOINTED) {
            fprintf(stderr, "process %d is locked; run 'checkpoint' to continue\n", pid);
        } else {
            fprintf(stderr, "process %d is already locked\n", pid);
        }
    } else {
        if (want == STATE_LOCKED) {
            fprintf(stderr, "process %d is checkpointed; run 'restore' to continue\n", pid);
        } else if (want == STATE_CHECKPOINTED) {
            fprintf(stderr, "process %d is already checkpointed\n", pid);
        } else {
            fprintf(stderr, "process %d is checkpointed; run 'resume' first\n", pid);
        }
    }
    return EXIT_USAGE;
}

static int do_get_state(int pid, int json)
{
    int state = 0;
    cu_result_t rc;
    char detail[32];
    unsigned long long t0;

    snprintf(detail, sizeof(detail), "pid=%d", pid);
    t0 = now_ms();
    rc = p_get_state(pid, &state);
    if (rc != 0 && rc == CUDA_ERROR_NOT_INITIALIZED && !driver_inited &&
        ensure_driver_init() == 0) {
        rc = p_get_state(pid, &state);
    }
    trace_call("cuCheckpointProcessGetState", detail, rc, t0);
    if (rc != 0) {
        fprintf(stderr, "failed to get state of process %d: %s\n",
                pid, error_string(rc));
        return 1;
    }
    if (json) {
        printf("{\"pid\": %d, \"state\": \"%s\"}\n", pid, state_name(state));
    } else {
        printf("%s\n", state_name(state));
    }
    return 0;
}

static int do_get_restore_tid(int pid, int json)
{
    int tid = 0;
    cu_result_t rc;
    char detail[32];
    unsigned long long t0;

    snprintf(detail, sizeof(detail), "pid=%d", pid);
    t0 = now_ms();
    rc = p_get_restore_tid(pid, &tid);
    if (rc != 0 && rc == CUDA_ERROR_NOT_INITIALIZED && !driver_inited &&
        ensure_driver_init() == 0) {
        rc = p_get_restore_tid(pid, &tid);
    }
    trace_call("cuCheckpointProcessGetRestoreThreadId", detail, rc, t0);
    if (rc != 0) {
        fprintf(stderr, "failed to get restore thread ID of process %d: %s\n",
                pid, error_string(rc));
        return 1;
    }
    if (json) {
        printf("{\"pid\": %d, \"tid\": %d}\n", pid, tid);
    } else {
        printf("%d\n", tid);
    }
    return 0;
}

static int do_toggle(int pid, gpu_pair_t *pairs, unsigned int count)
{
    int state = 0;
    cu_result_t rc;

    rc = p_get_state(pid, &state);
    if (rc != 0) {
        fprintf(stderr, "failed to get state of process %d: %s\n",
                pid, error_string(rc));
        return 1;
    }

    if (state == STATE_RUNNING) {
        if (do_lock(pid, 0) != 0) {
            return 1;
        }
        if (do_checkpoint(pid) != 0) {
            fprintf(stderr, "process %d is now locked; run 'checkpoint' to retry or 'unlock' to abort\n", pid);
            return 1;
        }
        printf("checkpointed\n");
        return 0;
    }
    if (state == STATE_CHECKPOINTED) {
        if (do_restore(pid, pairs, count) != 0) {
            return 1;
        }
        if (do_unlock(pid) != 0) {
            fprintf(stderr, "process %d is now locked; run 'unlock' to finish\n", pid);
            return 1;
        }
        printf("running\n");
        return 0;
    }
    if (state == STATE_LOCKED) {
        fprintf(stderr, "process %d is locked; run 'checkpoint' to continue or 'unlock' to abort\n", pid);
        return EXIT_USAGE;
    }
    if (state == STATE_FAILED) {
        fprintf(stderr, "process %d is in the failed state; it cannot be recovered\n", pid);
        return EXIT_USAGE;
    }
    fprintf(stderr, "process %d is %s; wait for the operation to finish\n",
            pid, state_name(state));
    return EXIT_USAGE;
}

static int do_save(int pid)
{
    if (state_guard(pid, STATE_RUNNING, "save") != 0) {
        return EXIT_USAGE;
    }
    if (do_lock(pid, 0) != 0) {
        return 1;
    }
    if (do_checkpoint(pid) != 0) {
        fprintf(stderr, "process %d is now locked; run 'checkpoint' to retry or 'unlock' to abort\n", pid);
        return 1;
    }
    printf("checkpointed\n");
    return 0;
}

static int do_resume(int pid, gpu_pair_t *pairs, unsigned int count)
{
    if (state_guard(pid, STATE_CHECKPOINTED, "resume") != 0) {
        return EXIT_USAGE;
    }
    if (do_restore(pid, pairs, count) != 0) {
        return 1;
    }
    if (do_unlock(pid) != 0) {
        fprintf(stderr, "process %d is now locked; run 'unlock' to finish\n", pid);
        return 1;
    }
    printf("running\n");
    return 0;
}

static int do_gpus(void)
{
    int count = 0;
    int i;
    cu_result_t rc;

    if (p_cu_device_get_count == NULL) {
        fprintf(stderr, "failed to list GPUs: driver does not export cuDeviceGetCount\n");
        return 1;
    }
    rc = p_cu_device_get_count(&count);
    if (rc != 0) {
        fprintf(stderr, "failed to get GPU count: %s\n", error_string(rc));
        return 1;
    }
    printf("index  uuid\n");
    for (i = 0; i < count; i++) {
        cu_uuid_t uuid;
        char text[48];

        if (p_cu_device_get_uuid == NULL ||
            p_cu_device_get_uuid(&uuid, i) != 0) {
            printf("%-5d  (error)\n", i);
            continue;
        }
        format_uuid(&uuid, text);
        printf("%-5d  GPU-%s\n", i, text);
    }
    return 0;
}

static int do_scan(int json)
{
    DIR *dir = opendir("/proc");
    struct dirent *ent;
    int found = 0;

    if (dir == NULL) {
        fprintf(stderr, "failed to open /proc: %s\n", strerror(errno));
        return 1;
    }
    if (json) {
        printf("[");
    } else {
        printf("%8s  %s\n", "pid", "state");
    }
    while ((ent = readdir(dir)) != NULL) {
        char *end = NULL;
        unsigned long value = strtoul(ent->d_name, &end, 10);
        int pid = (int)value;
        int state = 0;

        if (end == ent->d_name || *end != '\0' || value == 0 || value > 0x7fffffff) {
            continue;
        }
        if (query_state(pid, &state) != 0) {
            continue;
        }
        if (json) {
            printf("%s{\"pid\": %d, \"state\": \"%s\"}", found ? ", " : "", pid,
                   state_name(state));
        } else {
            printf("%8d  %s\n", pid, state_name(state));
        }
        found++;
    }
    closedir(dir);
    if (json) {
        printf("]\n");
    } else if (found == 0) {
        fprintf(stderr, "no CUDA processes found\n");
    }
    return 0;
}

int main(int argc, char **argv)
{
    static const struct option long_options[] = {
        { "pid", required_argument, NULL, 'p' },
        { "timeout", required_argument, NULL, 't' },
        { "device-map", required_argument, NULL, 'd' },
        { "json", no_argument, NULL, 'j' },
        { "trace", no_argument, NULL, 'T' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 },
    };
    int cmd = CMD_NONE;
    const char *pid_text = NULL;
    const char *timeout_text = NULL;
    const char *device_map = NULL;
    gpu_pair_t *pairs = NULL;
    unsigned int pair_count = 0;
    long timeout_ms = 0;
    int pid = 0;
    int json_on = 0;
    int c;
    int result;
    size_t i;
    char msg[160];

    if (argc > 1 && strcmp(argv[1], "--version") == 0) {
        print_version();
        return 0;
    }
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        help(stdout);
        return 0;
    }
    if (argc < 2) {
        return usage_error("no command given");
    }

    for (i = 0; i < sizeof(commands) / sizeof(commands[0]); i++) {
        if (strcmp(argv[1], commands[i].name) == 0) {
            cmd = commands[i].id;
            break;
        }
    }
    if (cmd == CMD_NONE) {
        snprintf(msg, sizeof(msg), "unknown command '%s'", argv[1]);
        return usage_error(msg);
    }

    optind = 1;
    while ((c = getopt_long(argc - 1, argv + 1, "+p:t:d:h", long_options, NULL)) != -1) {
        switch (c) {
        case 'p':
            pid_text = optarg;
            break;
        case 't':
            timeout_text = optarg;
            break;
        case 'd':
            device_map = optarg;
            break;
        case 'j':
            json_on = 1;
            break;
        case 'T':
            trace_on = 1;
            break;
        case 'h':
            cmd_help(stdout, cmd);
            return 0;
        default:
            return usage_error(NULL);
        }
    }
    if (optind < argc - 1) {
        snprintf(msg, sizeof(msg), "unexpected argument '%s'", argv[optind + 1]);
        return usage_error(msg);
    }

    if (timeout_text != NULL && cmd != CMD_LOCK) {
        return usage_error("--timeout is only supported by the lock command");
    }
    if (device_map != NULL && cmd != CMD_RESTORE && cmd != CMD_TOGGLE &&
        cmd != CMD_RESUME) {
        return usage_error("--device-map is only supported by the restore, resume and toggle commands");
    }
    if (json_on && cmd != CMD_STATE && cmd != CMD_RESTORE_TID && cmd != CMD_SCAN) {
        return usage_error("--json is only supported by the state, restore-tid and scan commands");
    }
    if (cmd == CMD_GPUS || cmd == CMD_SCAN) {
        if (pid_text != NULL) {
            return usage_error("the gpus and scan commands do not take --pid");
        }
    } else {
        if (pid_text == NULL) {
            return usage_error("missing --pid");
        }
        if (parse_pid(pid_text, &pid) != 0) {
            return usage_error("--pid must be a non-negative integer");
        }
    }
    if (timeout_text != NULL) {
        char *end = NULL;
        long value = strtol(timeout_text, &end, 10);

        if (end == timeout_text || *end != '\0' || value < 0) {
            return usage_error("--timeout must be a non-negative integer");
        }
        timeout_ms = value;
    }

    load_driver();

    if (cmd != CMD_STATE && cmd != CMD_RESTORE_TID) {
        if (ensure_driver_init() != 0) {
            return 1;
        }
    }
    if (device_map != NULL &&
        parse_device_map(device_map, &pairs, &pair_count) != 0) {
        return usage_error("invalid device map (each side must be a GPU index or a GPU-<hex> UUID)");
    }

    switch (cmd) {
    case CMD_STATE:
        result = do_get_state(pid, json_on);
        break;
    case CMD_SAVE:
        result = do_save(pid);
        break;
    case CMD_RESUME:
        result = do_resume(pid, pairs, pair_count);
        break;
    case CMD_LOCK:
        if (state_guard(pid, STATE_RUNNING, "lock") != 0) {
            return EXIT_USAGE;
        }
        result = do_lock(pid, timeout_ms);
        break;
    case CMD_CHECKPOINT:
        if (state_guard(pid, STATE_LOCKED, "checkpoint") != 0) {
            return EXIT_USAGE;
        }
        result = do_checkpoint(pid);
        break;
    case CMD_RESTORE:
        if (state_guard(pid, STATE_CHECKPOINTED, "restore") != 0) {
            return EXIT_USAGE;
        }
        result = do_restore(pid, pairs, pair_count);
        break;
    case CMD_UNLOCK:
        if (state_guard(pid, STATE_LOCKED, "unlock") != 0) {
            return EXIT_USAGE;
        }
        result = do_unlock(pid);
        break;
    case CMD_TOGGLE:
        result = do_toggle(pid, pairs, pair_count);
        break;
    case CMD_RESTORE_TID:
        result = do_get_restore_tid(pid, json_on);
        break;
    case CMD_GPUS:
        result = do_gpus();
        break;
    case CMD_SCAN:
        result = do_scan(json_on);
        break;
    default:
        result = 1;
        break;
    }

    return result;
}
