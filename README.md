# gpucheckpoint

gpucheckpoint saves and restores the GPU state of a running Linux process.
You can then move, pause, or fully checkpoint the process with
[CRIU](https://criu.org/Main_Page).

It is an independent open source (MIT) implementation of the same workflow
as NVIDIA's `cuda-checkpoint` tool. This project is not affiliated with
NVIDIA.

## Quick start

You need Linux and an NVIDIA driver, version 570 or newer.
The CUDA toolkit is not needed.

```sh
make
./gpucheckpoint state -p <pid>
# prints: running
```

`make` builds three files:

- `gpucheckpoint` - the command-line tool
- `libgpucheckpoint.so` - the same operations as a library (see below)
- `demo/demo` - a small example that uses the library; pass it a CUDA
  program to run, such as a build of `tests/counter.c`

## How it works

The NVIDIA GPU driver does all the checkpoint work. This tool is a small
client that calls the **public** driver API (`cuCheckpointProcess*`).
It finds these functions in `libcuda.so.1` at runtime.

- No NVIDIA code, text, or documentation is included.
- The NVIDIA driver is still required, like for every CUDA program.
- NVIDIA's official tool is a thin client over the same API, so both
  tools do the same for the core checkpoint workflow (see Limitations).

## Usage

The process moves through four states:

```
running -> lock -> locked -> checkpoint -> checkpointed
checkpointed -> restore -> locked -> unlock -> running
```

Commands:

| Command | Effect |
|---|---|
| `state -p <pid>` | Print the current state |
| `restore-tid -p <pid>` | Print the CUDA restore thread ID |
| `lock -p <pid> [-t <ms>]` | Lock the process |
| `checkpoint -p <pid>` | Move GPU memory to host memory, release GPU resources |
| `restore -p <pid> [-d <map>]` | Put the memory back on the GPU |
| `unlock -p <pid>` | Unlock the process |
| `save -p <pid>` | `lock` + `checkpoint` in one call |
| `resume -p <pid> [-d <map>]` | `restore` + `unlock` in one call |
| `toggle -p <pid> [-d <map>]` | `save` or `resume`, depending on the state |
| `gpus` | List the GPUs with their index and UUID |
| `scan` | Show the state of every CUDA process on the machine |

Options:

- `-p, --pid <pid>` - process ID. Required by the process commands.
- `-t, --timeout <ms>` - `lock` only. Wait up to `<ms>` milliseconds.
  On timeout, the process keeps its old state.
- `-d, --device-map <map>` - `restore`, `resume`, `toggle`. Format:
  `old=new` pairs, comma separated. Each side is a GPU index (`0`, `1`,
  ...) or a UUID (`GPU-<32 hex digits>`, with or without the dashes
  nvidia-smi prints). Indexes follow CUDA device order (as shown by
  `gpus`), so `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` change them.
  It must name every checkpointed GPU.
  The target GPU must be the same chip type with enough memory.
  (Driver 580 or newer.)
- `--json` - `state`, `restore-tid`, `scan`. Machine-readable output.
- `--trace` - print each driver call with its result and duration.
- `-h, --help` - help for a command. `--version` - version and driver.

Notes:

- States are: `running`, `locked`, `checkpointed`, `failed`. Newer
  drivers may briefly report `checkpointing` or `restoring`.
- Running a command in the wrong state exits with code `2` and a message
  that names the command to run next.
- Exit codes: `0` success, `1` runtime error, `2` usage error.
- The target process must belong to the calling user, or run as root.

### Full process checkpoint with CRIU

```sh
gpucheckpoint save -p $PID
criu dump --shell-job --images-dir demo --tree $PID
# later, on the same or different hardware:
criu restore --shell-job --restore-detached --images-dir demo
gpucheckpoint resume -p $PID
```

## In-process library

If your program (a scheduler, a CRIU hook, a language binding) calls the
API itself, use `libgpucheckpoint.so` instead of starting the CLI.
Each call goes straight to the driver. No process is started.

```c
#include "gpucheckpoint.h"

gc_init();
gc_lock(pid, 0);
gc_checkpoint(pid);
/* ... criu dump ... */
gc_restore(pid, NULL, 0);
gc_unlock(pid);
```

All functions return `GC_OK` (0) on success, or an error code. Pass the
code to `gc_error_string()` to get the message.
`gc_parse_device_map()` converts the `-d` device-map format (UUID form)
into the struct that `gc_restore()` takes. See `demo/demo` for a working
example.

## Verification

Tested side by side with the official
[cuda-checkpoint@00d5cce](https://github.com/NVIDIA/cuda-checkpoint/commit/00d5cce84c628088d6caa203fc4af40c1538b6f7):

- The happy path (states, restore thread IDs, exit codes) matches the
  official tool.
- A process with a GPU counter that grows every second: the value stops
  during checkpoint and continues after restore. Samples from the start,
  middle, and end of the buffer are equal after a full cycle.
- Lock timeout on a spinning kernel: both tools fail with the driver's
  "device not ready" error, and the process stays `running`.
- A simple UVM (managed memory) allocation checkpointed fine. However,
  NVIDIA's docs still list UVM as unsupported; test your workload.

## Limitations

- `--launch-job` (the 610+ IPC job feature) is not implemented. Use the official tool for this.
- Driver 570 or newer is required. (The official tool also works on
  550/560 through a private path.)
- Driver-side limits, same as the official tool:
  - Pending GPU work must finish before a checkpoint completes.
  - IPC memory from `cuMemExportToShareableHandle()` is not supported
    on driver 610 and below.
  - If a checkpoint fails, the process enters the `failed` state.

## Testing

```sh
make test
```

The suite starts a small GPU app (`tests/counter.c`) and checks the state
machine (low-level and `save`/`resume`), error paths, `gpus`/`scan`,
JSON and trace output, the library, and GPU-value continuity.
It skips itself when no NVIDIA driver is present.

To compare the happy-path output with the official binary:

```sh
OFFICIAL=/path/to/official/cuda-checkpoint make test
```

## License

MIT - see [LICENSE](LICENSE).
