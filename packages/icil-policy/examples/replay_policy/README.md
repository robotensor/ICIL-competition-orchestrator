# replay_policy

A complete ICIL competition submission that replays the demonstration it is given: the k-th call
to `act` after `reset` returns the demonstration's `actions[k]`, and the last action once the
demonstration has run out. It ignores observations, so it is the floor any learned policy has to
beat.

A submission repository holds:

- `icil.yaml` - names the policy class (`replay.policy:ReplayPolicy`) and its requirements;
- `requirements.txt` - what the policy needs beyond `icil-policy`;
- `replay/` - the policy's code, importable from the repository root.

Serve it locally, the way the competition does in a sandboxed container:

```bash
pip install icil-policy   # or: pip install -e packages/icil-policy from the orchestrator repo
export ICIL_POLICY_AUTHKEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python -m icil_policy.serve --manifest icil.yaml --address /tmp/replay.sock \
    --authkey-env ICIL_POLICY_AUTHKEY --log-file /tmp/replay.log
```

and drive it from another shell:

```python
import os
import numpy as np
from icil_policy.client import RemotePolicy

T, D = 50, 16
demo = {"qpos": np.zeros((T, D)), "actions": np.random.rand(T - 1, D)}
with RemotePolicy("/tmp/replay.sock", bytes.fromhex(os.environ["ICIL_POLICY_AUTHKEY"])) as policy:
    print(policy.hello())
    policy.set_demonstration(demo, {"frequency": 25.0})
    policy.reset(seed=0)
    assert np.array_equal(policy.act({"qpos": np.zeros(D)})["action"], demo["actions"][0])
```
