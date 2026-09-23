"""Robotensor Vector's policy protocol: a policy served in its own process, spoken to over named arrays.

A benchmark drives a policy it cannot import - the competitor's stack pins what the simulator pins,
and a submission is untrusted code - so the two meet on a socket. This distribution is both ends of
that socket and nothing else: it knows no benchmark, no channel name and no simulator.

- `Policy`: what a competitor implements.
- `vector_policy.manifest`: `icil.yaml`, which names the policy in a competitor's repository.
- `python -m vector_policy.serve`: serves that policy to one client.
- `vector_policy.client.RemotePolicy`: the client a benchmark drives it with.
- `vector_policy.wire`: the message format between them.
"""

from .errors import ManifestError, PolicyUnavailable, WireError
from .policy import ACTION_TYPES, Policy

__version__ = "0.1.0.dev0"

__all__ = [
    "ACTION_TYPES",
    "ManifestError",
    "Policy",
    "PolicyUnavailable",
    "WireError",
    "__version__",
]
