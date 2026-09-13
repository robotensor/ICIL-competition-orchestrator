"""The duel's runtime seam over the submission sandbox, with `FakeDocker` in Docker's place.

`FakeDocker.run` starts the real `python -m icil_policy.serve` on the host where the container
would be, so a whole duel runs through the adapter: fetch, manifest, image, health check and one
container per unit. `pytest -m container` runs the same duel with Docker itself.
"""

from __future__ import annotations

import os

import pytest

from conftest import fake_spec_doc
from duel_helpers import REPLAY, REPLAY_REF, ZERO, ZERO_REF, RecordingReporter
from icil_orchestrator.canon import Signer
from icil_orchestrator.duel.docker_runtime import DockerPolicyRuntime
from icil_orchestrator.duel.orchestrate import DuelRequest, Orchestrator
from icil_orchestrator.duel.runtime import PolicyRuntime, RuntimeUnavailable, SubmissionRefused
from icil_orchestrator.store.verify import verify_store
from icil_orchestrator.store.writer import Store
from icil_orchestrator.submissions.fetch import tree_hash
from store_helpers import make_record, publish
from submission_helpers import FAKE_BASE_DIGEST, FakeDocker

TRACK = "franka_1arm"


@pytest.fixture
def spec(spec_doc, write_spec, fake_installed):
    """The fake benchmark's track, with a sandbox user this process can hand a directory to."""
    doc = fake_spec_doc(spec_doc)
    doc["budgets"]["act_timeout_s"] = 2.0
    if os.getuid() != 0:
        doc["submission"]["sandbox"]["user"] = f"{os.getuid()}:{os.getgid()}"
    return write_spec(doc, name="docker-duel-spec.json")


@pytest.fixture
def docker():
    fake = FakeDocker()
    fake.images["icil-policy-base:latest"] = FAKE_BASE_DIGEST
    yield fake
    fake.kill_all()


class Killing(DockerPolicyRuntime):
    """Kills the container of the units named in `kill`, as soon as it listens."""

    kill: set[str] = set()

    def _started(self, container, served):
        if served.log_file.parent.name in self.kill:
            self.docker.remove(container.name)


def runtime_for(spec, docker, tmp_path, cls=DockerPolicyRuntime, **kwargs):
    return cls(
        spec,
        docker,
        cache_dir=tmp_path / "cache",
        local={REPLAY_REF.repo: REPLAY, ZERO_REF.repo: ZERO, **kwargs.pop("local", {})},
        base_digest=FAKE_BASE_DIGEST,
        gpus=0,
        **kwargs,
    )


def duel(spec, runtime, tmp_path):
    signer = Signer.generate()
    store = Store(tmp_path / "store", spec, signer)
    store.init(signer.verify_key_hex)
    publish(store, spec, make_record(spec, "genesis", 1, ZERO_REF, None))
    orchestrator = Orchestrator(
        spec, store, runtime, tmp_path / "runs", live=RecordingReporter(spec)
    )
    return store, orchestrator.run(DuelRequest(TRACK, REPLAY_REF, ZERO_REF, "smoke", block=2))


def test_a_duel_runs_through_the_sandbox_one_container_per_unit(spec, docker, tmp_path):
    runtime = runtime_for(spec, docker, tmp_path)
    assert isinstance(runtime, PolicyRuntime) and runtime.name == "docker"
    store, result = duel(spec, runtime, tmp_path)
    assert result.published and result.record["dethroned"], result.reason
    assert verify_store(store.root, spec).ok

    event = store.event(TRACK, result.event_id)
    for side, directory in (("challenger", REPLAY), ("king", ZERO)):
        recorded = event["sides"][side]
        assert recorded["commit"] == tree_hash(directory)
        assert recorded["base_image_digest"] == FAKE_BASE_DIGEST
        assert recorded["image"].startswith("sha256:") and recorded["runtime"] == "docker"
    # Two health checks, then one container for each unit of each side, every one removed.
    names = [args[args.index("--name") + 1] for args in docker.runs]
    assert len(names) == 2 + 2 * len(result.units)
    assert all(name.startswith("icil-duel-") for name in names)
    assert sorted(docker.removed) == sorted(names) and docker.processes == {}
    for args in docker.runs:
        assert args[args.index("--network") + 1] == "none"
        mounts = [a for a in args if a.startswith("type=bind")]
        assert len(mounts) == 1, "a unit's container mounted more than its socket directory"
        assert str(tmp_path) not in mounts[0], "the run directory or the store was mounted"
    for side in ("challenger", "king"):
        for unit in result.units:
            assert (result.run_dir / side / unit["unit_id"] / "policy.log").is_file()


def test_a_submission_that_does_not_build_is_refused_and_the_duel_void(spec, docker, tmp_path):
    docker.build_failure = "ERROR: No matching distribution found for numpy"
    runtime = runtime_for(spec, docker, tmp_path)
    store, result = duel(spec, runtime, tmp_path)
    assert result.status == "void"
    assert result.reason.startswith("the challenger's submission was refused: build: installing")
    assert docker.runs == [] and len(store.iter_index(TRACK)) == 1


def test_a_container_that_dies_under_its_unit_voids_the_rest_of_its_side(spec, docker, tmp_path):
    runtime = runtime_for(spec, docker, tmp_path, cls=Killing)
    runtime.kill = {"fp-000"}
    store, result = duel(spec, runtime, tmp_path)
    assert result.status == "void"
    assert "the policy container exited" in result.units[0]["challenger_error"]
    assert "died earlier in this side" in result.units[1]["challenger_error"]


def test_the_seams_errors_are_the_sandboxs_mapped(spec, docker, tmp_path):
    runtime = runtime_for(spec, docker, tmp_path, local={"org/missing": tmp_path / "nowhere"})
    with pytest.raises(RuntimeUnavailable, match="is not a directory"):
        runtime.resolve("org/missing", "main")
    (tmp_path / "empty").mkdir()
    runtime.local["org/empty"] = tmp_path / "empty"
    fetched = runtime.fetch(runtime.resolve("org/empty", "main"), workdir=tmp_path)
    with pytest.raises(SubmissionRefused) as refused:
        runtime.prepare(fetched, workdir=tmp_path)
    assert refused.value.step == "manifest"
