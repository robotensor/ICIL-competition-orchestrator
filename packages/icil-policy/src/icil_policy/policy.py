"""What a competitor implements: the `Policy` protocol.

A policy is a plain class. The server builds it once from `icil.yaml`, with the manifest's
`kwargs`, and then, for as long as its one client is connected:

- `set_demonstration(arrays, info)` hands it the one demonstration, as named arrays. The names are
  the benchmark's (RoboTwin: `frames_<camera>`, `qpos`, `endpose`, `actions`, `times`,
  `frequency`); `info` carries public fields only, such as the frame rate and the camera names.
  Privileged prompt metadata never reaches a policy.
- `reset(seed)` starts an episode.
- `act(observation)` answers one observation with at least `{"action": ...}`, one action of shape
  `(A,)` or a chunk of shape `(H, A)`, in the space `action_type` names.
- `close()`, if the policy has it, is called when the client says `close`.

Arrays arrive read-only: copy one before changing it in place. Arrays go back as bool, integer or
float numpy arrays; anything else, object arrays above all, is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

#: The action spaces a policy may declare: joint positions or end-effector poses.
ACTION_TYPES = ("qpos", "ee")


@runtime_checkable
class Policy(Protocol):
    """A policy the server can serve. `close(self) -> None` is optional and not part of the check."""

    #: `"qpos"` or `"ee"`, sent to the client in the reply to `hello`.
    action_type: str

    def reset(self, seed: int) -> None: ...

    def set_demonstration(
        self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]
    ) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]: ...
