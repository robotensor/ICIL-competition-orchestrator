# The base of every submission image (spec.json `submission.base_image`): a CUDA runtime,
# Python 3.10 and icil-policy, and nothing of the orchestrator. The build context is the
# repository root:
#
#     icil-orchestrator submission build-base        # builds it and prints its digest
#
# A submission's image is built FROM this image by digest, so the same code runs on the same
# base wherever it is judged; the digest is pinned in spec.json and named by every event record.
#
# Pinned by digest too: the same Dockerfile at the same tag on another day is not the same base.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu 22.04's python3 is 3.10, the version the protocol is tested on.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

# The user the sandbox runs as (spec.json `submission.sandbox.user`; the build passes it). Its
# home is /tmp, the one writable place inside a read-only container.
ARG POLICY_UID=1000
ARG POLICY_GID=1000
RUN groupadd --gid "${POLICY_GID}" policy \
    && useradd --uid "${POLICY_UID}" --gid "${POLICY_GID}" --home-dir /tmp --no-create-home \
       --shell /usr/sbin/nologin policy

# The policy protocol, as the competitor's side serves it. Installed from this checkout, not from
# an index, so the base holds exactly the protocol the orchestrator was built with.
COPY packages/icil-policy /opt/icil-policy
RUN python -m pip install /opt/icil-policy && rm -rf /opt/icil-policy

# No entrypoint and no command: the orchestrator says what runs (python -m icil_policy.serve ...),
# and a submission's image adds only its files and its requirements.
ENTRYPOINT []
CMD []
WORKDIR /submission
