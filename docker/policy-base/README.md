# The policy base image

What every submission's image is built `FROM`: CUDA 12.8.1's `devel` image on Ubuntu 22.04 (pinned
by digest in the Dockerfile), a C toolchain, Python 3.10 with its headers, the sandbox user from
`spec.json` and `icil-policy` installed from this checkout. No entrypoint and no command: the
orchestrator runs `python -m icil_policy.serve` in it, and a submission's image adds only its
checkout at `/submission` and a `pip install -r` of its requirements.

## What is installed, and why

| what | why |
| --- | --- |
| `nvidia/cuda:12.8.1-devel-ubuntu22.04` | CUDA 12.8 is the first toolkit for the RTX 5090's Blackwell GPUs; `devel` rather than `runtime` for nvcc and the CUDA headers, which Triton, `torch.utils.cpp_extension` and any other kernel built at run time need |
| `python3`, `python3-pip`, `python3-venv` | Ubuntu 22.04's Python is 3.10, the version the protocol is tested on |
| `build-essential` (gcc, g++, make) | the compilers torch.compile's CPU backend, Triton's launcher, cffi and C++ extensions call |
| `python3.10-dev` | `Python.h`, which they compile against |
| `icil-policy` | the protocol's server |

Nothing else is added. Whatever a policy needs beyond that - torch, its weights' libraries - its
requirements install into its own image.

## Building and pinning

```bash
icil-orchestrator submission build-base            # context: the repository root
```

prints the image's digest (`sha256:<64 hex>`, the id Docker computes over its configuration and
layers) and tags the image `icil-policy-base:<hex>` and `icil-policy-base:latest`. The digest
belongs in `spec.json` under `submission.base_image.digest`; until it is pinned there, `submission
check --base-image <digest>` names it, and every check or duel record names the digest it ran on.

## The digest is not reproducible

Every build of this Dockerfile that is not wholly Docker's cache gets a digest of its own, from
the same Dockerfile and the same checkout. The image id hashes the image's configuration, which
records when each layer was made. The layers carry file times and whatever `apt-get` fetched that
day. On the development host three builds of the `runtime` base this one replaced gave three
digests: `5917dd63...` and `b3016d3c...` differ from the `apt-get` layer on, and `c73a73c2...`
from the copy of `packages/icil-policy`, whose `.pytest_cache` a test run had rewritten a minute
before. So:

- No digest written here, the one below included, is the base. It is one build's, an example.
- The digest that counts is that of the image you built, read on the host that has it:

  ```bash
  icil-orchestrator submission build-base                  # prints it on stdout
  icil-orchestrator submission build-base --json           # "image_id"
  docker image inspect --format '{{.Id}}' icil-policy-base:latest   # an image built before
  ```

- Pinning a digest pins that one image. A host that judges against it needs that image, moved with
  `docker save` and `docker load` (which keep its id) or a registry, never a rebuild. A rebuilt
  base is another digest, and `submission check` refuses a digest that is not on the host.
- Whatever digest a check or a duel ran on is the one its record names: `SubmissionReport`'s
  `base_image`, and the `base_image` of the side it gives an event.

BuildKit resolves `FROM name@sha256:...` against a registry's manifest digest, which a locally
built image does not have, and refuses a bare image id; so the base is reached through the tag
named by its digest, and the id behind that tag is checked against the digest immediately before
every submission build (`icil_orchestrator.submissions.image.ensure_base`).

## What a policy finds at run time

- `/submission`: its checkout, copied into the image at build time and read-only like the rest of
  the root filesystem. (Issue #4's scope said the repository would be bind-mounted read-only; it is
  copied instead, so that the requirements can install from it, the image id recorded with a side
  is the code that ran, and nothing of the host but the socket directory is mounted.)
- `/tmp`: a tmpfs of `sandbox.tmpfs_bytes` (8 GiB), mounted `nosuid,nodev` and, as
  `sandbox.tmpfs_exec` says, `exec` - the only writable place of its own. It is empty at every
  start: the container makes `HOME` (`/tmp/home`) and `XDG_CACHE_HOME` (`/tmp/home/.cache`) before
  the server starts, and points `TMPDIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and
  `TORCH_EXTENSIONS_DIR` into it (`container.policy_environment`). So a policy may compile at run
  time - `torch.compile`, Triton, cffi, `torch.utils.cpp_extension`, its own `gcc -shared` - and
  load what it built. It is the only place it can: a compile into the root filesystem (`/`,
  `/submission`, `/usr`, `/opt`) meets the read-only root, and `/dev/shm` (the other tmpfs Docker
  gives a container) and `/run/icil` take the file but are `noexec`, so it will not load
  (`tests/test_submission_jit.py` walks every mount to check). Its contents count against
  `memory_bytes` and go with the container.
- `/run/icil`: the socket and the server's log, shared with the host. When the orchestrator runs as
  root it is a tmpfs of 64 MiB and 64 entries (`container.SHARED_DIR_BYTES`,
  `SHARED_DIR_INODES`) mounted `nosuid,nodev,noexec` (`SHARED_DIR_HARDENING`), which the bind
  mount into the container keeps. As another user it is a plain directory with no cap, mounted
  with whatever options the host's filesystem there has, so it may run code; with no capabilities
  and no new privileges, a setuid file or a device node there is still inert.
- No network, no swap (`memory_bytes` is the total), no capabilities, no setuid escalation, and the
  spec's cpu and pid limits.

**Why `/tmp` may run code.** Submission code already runs natively in its container, and a
checkout may ship compiled libraries of its own; letting it also run what it compiles into a
size-capped `nosuid,nodev` tmpfs removes a speed bump, not a boundary. The boundary is
`--network none`, the read-only root, the non-root user with no capabilities, and the memory, cpu
and pid limits, and none of them changed.

The image build, where the requirements install with network, is bounded by `submission check
--build-timeout` (1800 s by default). A build that fails or runs past it is followed by the
orchestrator's own probe (`image.probe_index`). The probe is a build from the base with an empty
context, never cached, in which pip downloads `pip`. If the probe gets through, the failure is
the submission's rejection at build, whatever its log says, since a `setup.py` can print pip's
network errors. If the probe cannot reach the index either, the failure is the harness's error,
to try again. Checked images stay
until `icil-orchestrator submission prune`, which removes every `icil-submission` image no
container was made from. BuildKit's own cache is shared with every other build on the host and
is left to `docker builder prune`.

## Measured on the development host

Docker 27.3.1, x86_64, one RTX 5090 (driver 580.173.02), 2026-09-14, from this Dockerfile at the
commit that builds it from `devel`:

| what | value |
| --- | --- |
| base image digest (that build's; an example, not the base) | `sha256:25b8d365d3ad01bfa3bbf460581d2925a95d60439e576e3ef6fdf7861b522765` |
| base image size | 9 488 012 261 bytes (9.49 GB; `nvidia/cuda:12.8.1-devel-ubuntu22.04` alone is 9 341 554 090 bytes, and the `runtime` base this replaced was 3.52 GB) |
| pulling the CUDA `devel` image | 72 s |
| cold build (CUDA image pulled, no cached layer) | 22 s |
| cached rebuild | 0.35 s |
| replay example: `submission check`, `--gpus 0` | image build 0.4 s; `docker run` to listening 0.56 s; `hello` answered 0.13 s after that |
| `tests/fixtures/jit_policy`: first `act` (write C, `gcc -shared`, `dlopen`) | 30 ms; the next `act` 0.5 ms |
| `tests/fixtures/torch_compile_policy` (torch 2.8.0+cpu): image build | 13.4 s; 10 263 597 002 bytes, torch adding 0.78 GB to the base |
| its `hello` (import torch, `torch.compile` one function with inductor, one compile worker) | 5.5 s, 4.4 s of it compiling |
| its `act` (the compiled function) | 1.1 ms round trip, 0.37 ms in the call |

The digest in the table is that one build's and no other build will have it (see
[The digest is not reproducible](#the-digest-is-not-reproducible)).
`budgets.policy_start_seconds` (600 s) is far above what a policy without weights needs, or one
that compiles a small function; what should size it is a policy loading a model on the GPU and
compiling its kernels, measured on the first smoke duel.
