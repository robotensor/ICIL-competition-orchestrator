# The policy base image

What every submission's image is built `FROM`: a CUDA 12.8 runtime on Ubuntu 22.04 (pinned by
digest in the Dockerfile), Python 3.10, the sandbox user from `spec.json` and `icil-policy`
installed from this checkout. No entrypoint and no command: the orchestrator runs
`python -m icil_policy.serve` in it, and a submission's image adds only its checkout at
`/submission` and a `pip install -r` of its requirements.

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
day. On the development host three builds gave three digests: `5917dd63...` and `b3016d3c...`
differ from the `apt-get` layer on, and `c73a73c2...` from the copy of `packages/icil-policy`,
whose `.pytest_cache` a test run had rewritten a minute before. So:

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
- `/tmp`: a tmpfs, the only writable place of its own, and the sandbox user's `$HOME`. Docker
  mounts it `noexec`, and the base has no C compiler: nothing written there can be run or
  `dlopen`ed (checked: a shared library copied to `/tmp` fails with "failed to map segment from
  shared object", an executable with "Permission denied"). A policy that compiles kernels at run
  time - triton, `torch.compile`, cupy's JIT - fails at `hello` or `act`; compile ahead, into the
  image, or do without. Its contents count against `memory_bytes`.
- `/run/icil`: the socket and the server's log, shared with the host. When the orchestrator runs as
  root it is a tmpfs of 64 MiB and 64 entries (`container.SHARED_DIR_BYTES`,
  `SHARED_DIR_INODES`); as another user it is a plain directory with no cap.
- No network, no swap (`memory_bytes` is the total), no capabilities, no setuid escalation, and the
  spec's cpu and pid limits.

The image build, where the requirements install with network, is bounded by `submission check
--build-timeout` (1800 s by default); a build past it is a rejection at build. Checked images stay
until `icil-orchestrator submission prune`, which removes every `icil-submission` image no
container was made from. BuildKit's own cache is shared with every other build on the host and
is left to `docker builder prune`.

## Measured on the development host

Docker 27.3.1, x86_64, one RTX 5090, 2026-09-13, from this Dockerfile at the commit that adds it:

| what | value |
| --- | --- |
| base image digest (that build's; an example, not the base) | `sha256:5917dd63b09291b37a1c7a644bcc780cd145d9ec179e54c58f13cf345cca4154` |
| base image size | 3 520 484 642 bytes (3.52 GB; the CUDA runtime alone is 3.40 GB) |
| cold build (`--no-cache`, CUDA image already pulled) | 17 s |
| cached rebuild | about 1 s |
| replay example's image: build | 1.4 s (the checkout, and a `pip install` that finds numpy already there); 3.52 GB, all but a few KB shared with the base |
| replay example: `docker run` to listening | 0.5 s |
| replay example: `hello` answered | 0.1 s after that |

The digest in the table is that one build's and no other build will have it (see
[The digest is not reproducible](#the-digest-is-not-reproducible)).
`budgets.policy_start_seconds` (600 s) is far above what a policy without weights needs;
what should size it is a policy loading a model on the GPU, measured on the first smoke duel.
