# zero_policy

A complete Robotensor Vector submission whose every action is zeros, shaped like one row of the
demonstration's `actions`. It shows the smallest repository the competition accepts:

- `icil.yaml` - names the policy class (`zero.policy:ZeroPolicy`), its constructor `kwargs` and
  its requirements;
- `requirements.txt` - what the policy needs beyond `vector-policy`;
- `zero/` - the policy's code, importable from the repository root.

Serve it with

```bash
export VECTOR_POLICY_AUTHKEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python -m vector_policy.serve --manifest icil.yaml --address 127.0.0.1:5555 \
    --authkey-env VECTOR_POLICY_AUTHKEY
```

and connect with `vector_policy.client.RemotePolicy("127.0.0.1:5555", authkey)`; see
`../replay_policy/README.md` for a client session.
