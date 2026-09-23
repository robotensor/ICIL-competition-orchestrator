# vector-policy

The policy protocol of Robotensor Vector, the subnet's action-grounded in-context learning
competition. A benchmark drives a policy it cannot import: the policy is served in its own process,
and the two exchange named numpy arrays and JSON fields over an authenticated socket. Nothing is
ever pickled, and the protocol knows no benchmark.

It depends on numpy and PyYAML only, because it is installed into every competitor's image. A
benchmark that only drives a policy imports `vector_policy.client`, which loads numpy and the
standard library and nothing else.

```bash
pip install -e packages/vector-policy     # from the orchestrator repository
pytest packages/vector-policy             # its tests, on their own
```

## For a competitor: a policy and its `icil.yaml`

A submission is a repository with `icil.yaml` at its root:

```yaml
api: 1                              # required
policy: my_policy.policy:MyPolicy   # required: module:Class, importable from the repository root
kwargs: {checkpoint: weights.pt}    # optional: passed to the constructor
requirements: requirements.txt      # optional: relative to the root, inside the repository
benchmarks: [robotwin]              # optional
```

Unknown or duplicated keys are refused; `vector_policy.manifest.load(path)` lists every problem in
one `ManifestError`. The class implements `vector_policy.Policy`:

```python
class MyPolicy:
    action_type = "qpos"  # or "ee"

    def set_demonstration(self, arrays, info): ...  # the one demonstration, as named arrays
    def reset(self, seed: int): ...  # a new episode
    def act(self, observation) -> dict: ...  # {"action": (A,) or (H, A) array, ...}
    def close(self): ...  # optional
```

The array names are the benchmark's. RoboTwin sends `frames_<camera>` `(T,H,W,3)` uint8, `qpos`
`(T,D)`, `endpose` `(T,E)`, `actions` `(T-1,D)`, `times` `(T,)` and `frequency` `()` as the
demonstration, with `info = {"frequency", "cameras", "embodiment", "action_dims"}`, and one
observation as `frames_<camera>` `(H,W,3)`, `qpos` `(D,)` and `endpose` `(E,)`. Privileged prompt
metadata never reaches a policy. Arrays arrive read-only.

Two complete repositories to copy: [`examples/replay_policy`](examples/replay_policy) replays the
demonstration's actions, [`examples/zero_policy`](examples/zero_policy) answers zeros.

## Serving

```bash
export VECTOR_POLICY_AUTHKEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python -m vector_policy.serve --manifest examples/replay_policy/icil.yaml \
    --address /tmp/policy.sock --authkey-env VECTOR_POLICY_AUTHKEY --log-file /tmp/policy.log
```

`--address` is a Unix socket path or `host:port`. The key, at least 16 bytes, is read from the
environment as hex and removed from it before competitor code runs. The server accepts one client,
builds the policy on its first `hello` (repository root first on `sys.path` and as the working
directory), and exits 0
after `close` or when the client hangs up - even in the middle of a call that never returns - or
says nothing for `--idle-timeout-s` (30 minutes by default; a call in progress is not idle). A
policy exception becomes an error reply and serving goes on; a malformed message gets an error
reply and ends the session (exit 1), as does a policy that cannot be built. Exit 2 means serving
never started.

## For a benchmark: `RemotePolicy`

```python
from vector_policy import PolicyUnavailable
from vector_policy.client import RemotePolicy

with RemotePolicy(address, authkey, timeout_s=60.0, log_file=server_log) as policy:
    policy.hello()  # {"protocol": 1, "action_type": "qpos", "policy": ...}
    policy.set_demonstration(arrays, info)
    policy.reset(seed)
    action = policy.act(observation)["action"]
```

Every failure raises `PolicyUnavailable` - an error reply, a call past `timeout_s`, a hang-up, a
refused key, a malformed reply - with the tail of `log_file` (or of the log the server sent) in its
message. Each call's timeout covers sending the request and receiving the whole reply; when it runs
out the connection is shut down and the server exits. After an error reply to `reset`, `prompt` or
`act` the policy can still be used; after any other failure, a failed `hello` included, the
connection is closed. `close` is best effort and raises nothing. An array the wire cannot carry is
the caller's mistake and raises `WireError` before anything is sent.

## The wire

`vector_policy.wire`, over `multiprocessing.connection` with an authkey, using `send_bytes` and
`recv_bytes` only. A message is one JSON header frame

```json
{"protocol": 1, "op": "act", "fields": {}, "arrays": [{"name": "qpos", "dtype": "<f8", "shape": [16]}]}
```

followed by one raw little-endian frame per array, in header order; a message holds at most 1024
arrays. Dtypes are bool, int8-64, uint8-64 and float16-64; object and every other dtype is refused
on send and on receive. Client ops: `hello` (`client`), `reset` (`seed`), `prompt` (arrays;
`info`), `act` (arrays), `close`. Replies: `ok` (after `hello`: `protocol`, `action_type`,
`policy`), `action` (arrays with `action`), `error` (`type`, `message`, `log_tail`).
