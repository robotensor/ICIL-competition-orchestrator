"""The duel's runtime seam over the submission sandbox, with `FakeDocker` in Docker's place.

`FakeDocker.run` starts the real `python -m icil_policy.serve` on the host where the container
would be, so a whole duel runs through the adapter: fetch, manifest, image, health check and one
container per unit. `pytest -m container` runs the same duel with Docker itself.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
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
from icil_orchestrator.duel.docker_runtime import DockerPolicyRuntime
from icil_orchestrator.duel.orchestrate import DuelRequest, Orchestrator
from icil_orchestrator.duel.runtime import (
    PolicyDied,
    PolicyRuntime,
    RuntimeUnavailable,
    SubmissionRefused,
)
from icil_orchestrator.store.verify import verify_store
from icil_orchestrator.store.writer import Store
from icil_orchestrator.submissions.container import AUTHKEY_ENV, Owner, run_argv
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


def test_a_duel_runs_through_the_sandbox_one_container_per_unit(
    spec, docker, tmp_path, shared_mounts
):
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
    shared_dirs = set()
    for args in docker.runs:
        mounts = [a for a in args if a.startswith("type=bind")]
        assert len(mounts) == 1, "a unit's container mounted more than its socket directory"
        shared = Path(mounts[0].removeprefix("type=bind,src=").split(",")[0])
        assert str(tmp_path) not in str(shared), "the run directory or the store was mounted"
        shared_dirs.add(shared)
        # The sandbox's own `docker run`, word for word - its scratch tmpfs, HOME and JIT caches,
        # the owner labels - with nothing added: owned by this process, which starts them.
        name, image = args[args.index("--name") + 1], args[args.index(AUTHKEY_ENV) + 1]
        assert args == run_argv(
            spec, image=image, name=name, socket_dir=shared, gpus=0, owner=Owner.current()
        )
    if os.geteuid() == 0:
        # Each socket directory was its own bounded tmpfs (recorded here, mounted for real under
        # `pytest -m container`).
        assert {call[1] for call in shared_mounts if call[0] == "mount"} == shared_dirs
    import json

    for side in ("challenger", "king"):
        for unit in result.units:
            unit_dir = result.run_dir / side / unit["unit_id"]
            assert (unit_dir / "policy.log").is_file()
            # The benchmark read a copy of the container's log beside its result, never the log
            # in the directory the container shares.
            given = json.loads((unit_dir / "given.json").read_text())["args"]
            assert given["policy_log"] == str(unit_dir / "policy-tail.log")
            assert "listening on" in (unit_dir / "policy-log-seen.txt").read_text()


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


def test_a_container_whose_orchestrator_was_killed_is_reaped_and_a_live_ones_is_not(
    spec, docker, tmp_path
):
    """`reap` is the sandbox's own reaping: a container whose owner process is gone is removed,
    whatever store it served, and one a live process holds - this one - is left running."""
    runtime = runtime_for(spec, docker, tmp_path)
    prepared = runtime.prepare(runtime.fetch(REPLAY_REF, workdir=tmp_path), workdir=tmp_path)
    # No process has a pid above PID_MAX_LIMIT (2**22): an orchestrator killed long ago.
    killed = Owner(2**22 + 1, "1", Owner.current().pidns)
    held, names = [], []
    for unit, owner in (("fp-000", Owner.current()), ("fp-001", killed)):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Owner, "current", classmethod(lambda cls, owner=owner: owner))
            serving = runtime.serve(prepared, workdir=tmp_path / unit)
            serving.__enter__()  # and never left
        held.append(serving)
        names.append(docker.runs[-1][docker.runs[-1].index("--name") + 1])
    alive, orphan = names

    restarted = runtime_for(spec, docker, tmp_path)
    assert restarted.reap() == [orphan]
    assert orphan in docker.removed and orphan not in docker.processes
    assert alive in docker.processes, "a live process's container was reaped"


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


class InspectHangs(DockerPolicyRuntime):
    """`docker inspect` stops answering about the container of the challenger's unit `unit`:
    while it starts, its process already dead (`starting`); once it listens, its process then
    killed (`killed`); or once it listens, its policy left to play the unit out (`alive`)."""

    unit = "fp-000"
    when = "starting"

    def _chosen(self, unit_dir: Path) -> bool:
        return unit_dir.name == self.unit and unit_dir.parent.name == "challenger"

    def serve(self, prepared, *, workdir):
        if self.when == "starting" and self._chosen(Path(workdir)):
            self.docker.hang_next_run = True
        return super().serve(prepared, workdir=workdir)

    def _started(self, container, served):
        if self.when == "starting" or not self._chosen(served.log_file.parent):
            return
        self.docker.hang.add(container.name)
        if self.when == "killed":
            self.docker.processes[container.name].kill()
            self.docker.processes[container.name].wait()


@pytest.mark.parametrize(
    "when, void, said",
    [
        ("starting", True, "could not be served: docker could not say whether"),
        ("killed", True, "docker could not say how the policy container ended"),
        ("alive", False, None),
    ],
)
def test_a_docker_inspect_that_times_out_voids_its_unit_and_never_undoes_a_score(
    spec, write_spec, docker, tmp_path, when, void, said
):
    """A Docker that does not answer `docker inspect` in time cannot say whose a unit's end was:
    the unit is void for the harness, the duel goes on to the next one, and nothing escapes the
    side. A unit already scored keeps its score."""
    import json

    doc = json.loads(spec.path.read_text())
    doc["duel"]["max_void_fraction"] = 0.5  # so that one void unit leaves the duel standing
    lenient = write_spec(doc, name="lenient-docker-duel-spec.json")
    runtime = runtime_for(lenient, docker, tmp_path, cls=InspectHangs)
    runtime.when = when
    store, result = duel(lenient, runtime, tmp_path)
    assert result.published, result.reason
    first, second, third = result.units
    assert first["void"] is void
    if void:
        error = first["challenger_error"]
        assert said in error and "did not finish within 60s" in error, error
    else:
        assert first["challenger_success"] is True, "a scored unit was undone"
    assert second["challenger_success"] is True and third["challenger_success"] is True
    assert result.record["void"] == int(void) and verify_store(store.root, spec).ok
