"""A smoke duel through the real sandbox: Docker, the base image and both example repositories.

`pytest -m container`. The examples need no GPU, so their containers get none (`--gpus 0`); the
fake benchmark runs on the host and reaches each unit's container through its socket directory.
Every container the duel starts is named `icil-duel-*` and removed when its unit is over; the
submission images built here are untagged at the end, by tag: an image id another image shares
is left alone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from duel_helpers import REPLAY, REPLAY_REF, ZERO, ZERO_REF
from icil_orchestrator.cli import main
from icil_orchestrator.duel.side import read_results
from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.store.writer import Store
from icil_orchestrator.submissions.docker import Docker, DockerError
from icil_orchestrator.submissions.fetch import tree_hash
from icil_orchestrator.submissions.image import build_base_image, submission_tag

pytestmark = pytest.mark.container

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK = "franka_1arm"


@pytest.fixture(scope="module")
def docker():
    client = Docker()
    try:
        client.image_id("hello-world:nonexistent")
    except DockerError as exc:
        pytest.skip(str(exc))
    if shutil.which("docker") is None:
        pytest.skip("no docker binary")
    return client


@pytest.fixture(scope="module")
def base(docker, spec):
    return build_base_image(docker, spec, REPO_ROOT)


def duel_containers() -> list[str]:
    done = subprocess.run(
        ["docker", "ps", "--all", "--filter", "name=icil-duel-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.split()


def test_a_smoke_duel_through_the_sandbox_publishes_the_replay_challenger_winning(
    docker, base, duel_spec, fake_installed, tmp_path, capsys
):
    before = set(duel_containers())
    root, key = tmp_path / "store", tmp_path / "keys" / "orchestrator.ed25519"
    spec = ["--spec", str(duel_spec.path)]
    assert main([*spec, "store", "init", str(root), "--key", str(key)]) == 0
    capsys.readouterr()
    common = [
        "--track",
        TRACK,
        "--size",
        "smoke",
        "--store",
        str(root),
        "--run-dir",
        str(tmp_path / "runs"),
        "--queue",
        str(tmp_path / "queue"),
        "--key",
        str(key),
        "--runtime",
        "docker",
        "--cache",
        str(tmp_path / "cache"),
        "--base-image",
        base.base.digest,
        "--gpus",
        "0",
        "--local",
        f"{REPLAY_REF.repo}={REPLAY}",
        "--local",
        f"{ZERO_REF.repo}={ZERO}",
    ]
    tags = [
        submission_tag(SubmissionRef.make(ref.repo, tree_hash(directory)))
        for ref, directory in ((REPLAY_REF, REPLAY), (ZERO_REF, ZERO))
    ]
    try:
        zero = f"{ZERO_REF.repo}@{ZERO_REF.revision}"
        assert main([*spec, "duel", "--challenger", zero, *common]) == 0, capsys.readouterr().err
        genesis = json.loads(capsys.readouterr().out)
        assert genesis["kind"] == "genesis" and genesis["status"] == "published"

        replay = f"{REPLAY_REF.repo}@{REPLAY_REF.revision}"
        assert main([*spec, "duel", "--challenger", replay, *common]) == 0, capsys.readouterr().err
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "published" and out["dethroned"] is True, out["reason"]

        store = Store(root, duel_spec)
        event = store.event(TRACK, out["event_id"])
        run_dir = Path(out["run_dir"])
        challenger = read_results(run_dir / "challenger")
        king = read_results(run_dir / "king")
        for unit in event["units"]:
            sha = unit["prompt_sha256"]
            assert challenger[unit["unit_id"]]["prompt_sha256"] == sha
            assert king[unit["unit_id"]]["prompt_sha256"] == sha
            assert (unit["challenger_success"], unit["king_success"]) == (True, False)
        for side in ("challenger", "king"):
            recorded = event["sides"][side]
            assert recorded["base_image_digest"] == base.base.digest
            assert recorded["image"].startswith("sha256:") and recorded["runtime"] == "docker"
            assert recorded["image"] in {docker.image_id(tag) for tag in tags}
        assert main([*spec, "store", "verify", str(root)]) == 0
        assert capsys.readouterr().out.strip().endswith("OK")
        assert set(duel_containers()) == before, "a duel's container outlived its unit"
    finally:
        for tag in tags:
            docker.remove_image(tag)
