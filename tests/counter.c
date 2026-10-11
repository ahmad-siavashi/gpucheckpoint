/*
 * counter.c - test application for the gpucheckpoint test suite.
 *
 * SPDX-License-Identifier: MIT
 *
 * Allocates an int in GPU memory, increments it once per second, and prints
 * the new value. Used to verify that GPU work is suspended while a process
 * is checkpointed and that GPU memory contents survive restore.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define DIE(x) \
    do { \
        CUresult r_ = (x); \
        if (r_ != CUDA_SUCCESS) { \
            fprintf(stderr, "FAIL %s -> %d\n", #x, (int)r_); \
            exit(1); \
        } \
    } while (0)

int main(void)
{
    CUcontext ctx;
    CUdeviceptr p;
    int v = 100;

    DIE(cuInit(0));
    DIE(cuDevicePrimaryCtxRetain(&ctx, 0));
    DIE(cuCtxSetCurrent(ctx));
    DIE(cuMemAlloc(&p, sizeof v));
    DIE(cuMemcpyHtoD(p, &v, sizeof v));

    for (;;) {
        int h;

        sleep(1);
        DIE(cuMemcpyDtoH(&h, p, sizeof h));
        h++;
        DIE(cuMemcpyHtoD(p, &h, sizeof h));
        printf("gpu value: %d\n", h);
        fflush(stdout);
    }
}
