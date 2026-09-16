# zero_policy

A complete ICIL competition submission whose every action is zeros, shaped like one row of the
demonstration's `actions`. It shows the smallest repository the competition accepts:

- `icil.yaml` - names the policy class (`zero.policy:ZeroPolicy`), its constructor `kwargs` and
  its requirements;
- `requirements.txt` - what the policy needs beyond `icil-policy`;
- `zero/` - the policy's code, importable from the repository root.

Serve it with

```bash
export ICIL_POLICY_AUTHKEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python -m icil_policy.serve --manifest icil.yaml --address 127.0.0.1:5555 \
    --authkey-env ICIL_POLICY_AUTHKEY
```

and connect with `icil_policy.client.RemotePolicy("127.0.0.1:5555", authkey)`; see
`../replay_policy/README.md` for a client session.
