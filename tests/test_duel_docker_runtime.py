"""The duel's runtime seam over the submission sandbox, with `FakeDocker` in Docker's place.

`FakeDocker.run` starts the real `python -m icil_policy.serve` on the host where the container
would be, so a whole duel runs through the adapter: fetch, manifest, image, health check and one
container per unit. `pytest -m container` runs the same duel with Docker itself.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest

from conftest import fake_spec_doc
from duel_helpers import (
    REPLAY,
    REPLAY_REF,
    ZERO,
    ZERO_REF,
    InspectingFakeDocker,
    RecordingReporter,
)
from icil_orchestrator.canon import Signer
from icil_orchestrator.duel.docker_runtime import STORE_LABEL, DockerPolicyRuntime
from icil_orchestrator.duel.orchestrate import DuelRequest, Orchestrator
from icil_orchestrator.duel.runtime import (
    PolicyDied,
    PolicyRuntime,
    RuntimeUnavailable,
    SubmissionRefused,
)
from icil_orchestrator.store.verify import verify_store
from icil_orchestrator.store.writer import Store
from icil_orchestrator.submissions.docker import ContainerState
from icil_orchestrator.submissions.fetch import tree_hash
from store_helpers import make_record, publish
from submission_helpers import FAKE_BASE_DIGEST

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
    fake = InspectingFakeDocker()
    fake.images["icil-policy-base:latest"] = FAKE_BASE_DIGEST
    yield fake
    fake.kill_all()


class Killing(DockerPolicyRuntime):
    """Ends the container of the challenger's units named in `kill` as soon as it listens:
    removed from outside (`remove`), its process killed (`exit`), or killed for memory (`oom`)."""

    kill: set[str] = set()
    how = "remove"

    def _started(self, container, served):
        unit_dir = served.log_file.parent
        if unit_dir.name not in self.kill or unit_dir.parent.name != "challenger":
            return
        if self.how == "remove":
            self.docker.remove(container.name)
            return
        if self.how == "oom":
            self.docker.oom_killed.add(container.name)
        process = self.docker.processes[container.name]
        process.kill()
        process.wait()


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
        assert f"{STORE_LABEL}={store.root.resolve()}" in args, "a container was not labelled"
        assert args[args.index("--network") + 1] == "none"
        mounts = [a for a in args if a.startswith("type=bind")]
        assert len(mounts) == 1, "a unit's container mounted more than its socket directory"
        assert str(tmp_path) not in mounts[0], "the run directory or the store was mounted"
    for side in ("challenger", "king"):
        for unit in result.units:
            assert (result.run_dir / side / unit["unit_id"] / "policy.log").is_file()


def test_a_challenger_that_does_not_build_is_refused(spec, docker, tmp_path):
    docker.build_failure = "ERROR: No matching distribution found for numpy"
    runtime = runtime_for(spec, docker, tmp_path)
    store, result = duel(spec, runtime, tmp_path)
    assert result.status == "refused"
    assert result.reason.startswith("the challenger's submission was refused: build: installing")
    assert docker.runs == [] and len(store.iter_index(TRACK)) == 1


def test_a_container_removed_from_outside_voids_its_unit_and_the_next_one_runs(
    spec, write_spec, docker, tmp_path
):
    import json

    doc = json.loads(spec.path.read_text())
    doc["duel"]["max_void_fraction"] = 0.5  # so that one void unit leaves the duel standing
    lenient = write_spec(doc, name="lenient-docker-duel-spec.json")
    runtime = runtime_for(lenient, docker, tmp_path, cls=Killing)
    runtime.kill, runtime.how = {"fp-000"}, "remove"
    store, result = duel(lenient, runtime, tmp_path)
    first, second, third = result.units
    assert first["void"] and "the policy container is gone" in first["challenger_error"]
    assert second["challenger_success"] is True, "a removed container stopped its side"
    assert third["challenger_success"] is True
    assert result.published and result.record["void"] == 1, result.reason


@pytest.mark.parametrize(
    "how, said", [("exit", "exited (-9)"), ("oom", "was killed for going over its")]
)
def test_a_container_that_ends_by_its_own_doing_fails_its_unit(spec, docker, tmp_path, how, said):
    runtime = runtime_for(spec, docker, tmp_path, cls=Killing)
    runtime.kill, runtime.how = {"fp-000"}, how
    store, result = duel(spec, runtime, tmp_path)
    assert result.published, result.reason
    first, second, third = result.units
    assert (first["challenger_success"], first["void"]) == (False, False)
    assert said in first["challenger_error"]
    assert second["challenger_success"] is True and third["challenger_success"] is True
    assert result.record["void"] == 0 and verify_store(store.root, spec).ok


def test_a_container_a_killed_orchestrator_left_is_reaped_by_the_next_one_of_its_store(
    spec, docker, tmp_path
):
    held, names = [], []
    for store in ("store", "other-store"):
        runtime = runtime_for(spec, docker, tmp_path)
        runtime.bind(store=tmp_path / store, runs=tmp_path / "runs")
        fetched = runtime.fetch(REPLAY_REF, workdir=tmp_path / store)
        prepared = runtime.prepare(fetched, workdir=tmp_path / store / "check")
        serving = runtime.serve(prepared, workdir=tmp_path / store / "fp-000")
        serving.__enter__()  # and never left: its orchestrator was killed
        held.append(serving)
        run = docker.runs[-1]
        assert f"{STORE_LABEL}={(tmp_path / store).resolve()}" in run
        names.append(run[run.index("--name") + 1])

    restarted = runtime_for(spec, docker, tmp_path)
    restarted.bind(store=tmp_path / "store", runs=tmp_path / "runs")
    assert restarted.reap() == [names[0]]
    assert names[0] in docker.removed and names[0] not in docker.processes
    assert names[1] in docker.processes, "another store's container was reaped"


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


def test_only_a_socket_itself_counts_as_a_units_policy_listening(spec, docker, tmp_path):
    """The policy writes in its socket's directory, so a link at the socket's name - even to a
    socket that listens - is not its policy listening: the unit waits for a socket itself, and
    fails its side when none comes in time."""
    runtime = runtime_for(spec, docker, tmp_path)
    runtime.docker = SimpleNamespace(state=lambda name: ContainerState(True, None))
    runtime.start_timeout_s = 0.3
    container = SimpleNamespace(name="icil-duel-link", socket_path=tmp_path / "policy.sock")
    elsewhere, served = socket.socket(socket.AF_UNIX), socket.socket(socket.AF_UNIX)
    try:
        elsewhere.bind(str(tmp_path / "elsewhere.sock"))
        elsewhere.listen()
        container.socket_path.symlink_to(tmp_path / "elsewhere.sock")
        with pytest.raises(PolicyDied, match="did not listen within 0.3s"):
            runtime._wait_listening(container)
        container.socket_path.unlink()
        served.bind(str(container.socket_path))
        runtime._wait_listening(container)
    finally:
        elsewhere.close()
        served.close()
