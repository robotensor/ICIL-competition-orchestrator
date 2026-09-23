# The base of every submission image (spec.json `submission.base_image`): the CUDA toolkit with
# its compiler, a C toolchain, Python 3.10 with its headers and vector-policy, and nothing of the
# orchestrator. The build context is the repository root:
#
#     vector-orchestrator submission build-base        # builds it and prints its digest
#
# A submission's image is built FROM this image by digest, so the same code runs on the same
# base wherever it is judged; the digest is pinned in spec.json and named by every event record.
#
# Pinned by digest too: the same Dockerfile at the same tag on another day is not the same base.
# The tag is nvidia/cuda:12.8.1-devel-ubuntu22.04 (the digest is its multi-platform index's).
# CUDA 12.8 is the first toolkit for the RTX 5090's Blackwell GPUs, and `devel` rather than
# `runtime` brings nvcc and the CUDA headers, for a policy that compiles kernels at run time.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu 22.04's python3 is 3.10, the version the protocol is tested on. build-essential (gcc, g++,
# make) and python3.10-dev (Python.h) are what compiling at run time needs: torch.compile's CPU
# backend, Triton's launcher, cffi and torch.utils.cpp_extension call the C or C++ compiler and
# include Python's headers. The sandbox's /tmp may run what they write there.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
       build-essential python3.10-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

# The user the sandbox runs as (spec.json `submission.sandbox.user`; the build passes it). Its
# home is /tmp/home, on the tmpfs that is the one writable place inside a read-only container:
# the container makes it at start and names it in HOME.
ARG POLICY_UID=1000
ARG POLICY_GID=1000
RUN groupadd --gid "${POLICY_GID}" policy \
    && useradd --uid "${POLICY_UID}" --gid "${POLICY_GID}" --home-dir /tmp/home --no-create-home \
       --shell /usr/sbin/nologin policy

# The policy protocol, as the competitor's side serves it. Installed from this checkout, not from
# an index, so the base holds exactly the protocol the orchestrator was built with.
COPY packages/vector-policy /opt/vector-policy
RUN python -m pip install /opt/vector-policy && rm -rf /opt/vector-policy

# No entrypoint and no command: the orchestrator says what runs (python -m vector_policy.serve ...),
# and a submission's image adds only its files and its requirements.
ENTRYPOINT []
CMD []
WORKDIR /submission
