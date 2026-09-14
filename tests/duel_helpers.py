"""What the duel tests share: the example policies, a runtime with faults, a recording reporter,
a benchmark that voids chosen units for the harness, and a `FakeDocker` that answers `inspect`."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from icil_orchestrator.duel.local_runtime import SubprocessPolicyRuntime
from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.live import LiveReporter
from submission_helpers import FakeDocker

EXAMPLES = Path(__file__).resolve().parents[1] / "packages" / "icil-policy" / "examples"
REPLAY = EXAMPLES / "replay_policy"
ZERO = EXAMPLES / "zero_policy"

REPLAY_REF = SubmissionRef.make("robotensor/icil-replay-policy", "2" * 40)
ZERO_REF = SubmissionRef.make("robotensor/icil-zero-policy", "1" * 40)


class Crash(BaseException):
    """Stands for the orchestrator being killed: nothing in a duel catches it."""


class FakePolicyRuntime(SubprocessPolicyRuntime):
    """The subprocess runtime, with the ways a policy runtime goes wrong on demand.

    `kill_on_serve`: serve numbers (0-based, counted across both sides) whose server is killed as
    soon as it listens, the way a container dies. `crash_on_serve`: the serve number at which the
    orchestrator itself "dies" (`Crash`), before the unit runs.
    """

    def __init__(
        self,
        spec: Any,
        local: dict[str, Path] | None = None,
        *,
        kill_on_serve: set[int] | None = None,
        kill_repo: str | None = None,
        crash_on_serve: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            spec,
            local if local is not None else {REPLAY_REF.repo: REPLAY, ZERO_REF.repo: ZERO},
            start_timeout_s=kwargs.pop("start_timeout_s", 30.0),
            **kwargs,
        )
        self.kill_on_serve = set(kill_on_serve or ())
        self.kill_repo = kill_repo
        self.crash_on_serve = crash_on_serve
        #: `(repo, unit directory name)` for every unit served.
        self.serves: list[tuple[str, str]] = []
        self.prepared: list[str] = []
        #: Which serve is starting, or None for the health check `prepare` runs.
        self._number: int | None = None
        self._current: str | None = None

    def prepare(self, fetched, *, workdir):
        self.prepared.append(fetched.ref.repo)
        self._number = None
        return super().prepare(fetched, workdir=workdir)

    def serve(self, prepared, *, workdir):
        number = len(self.serves)
        if number == self.crash_on_serve:
            self.crash_on_serve = None
            raise Crash(f"the orchestrator died before serving unit {Path(workdir).name}")
        self.serves.append((prepared.ref.repo, Path(workdir).name))
        self._current = prepared.ref.repo
        self._number = number
        return super().serve(prepared, workdir=workdir)

    def _started(self, process, served):
        if self._number is None:
            return
        killed = self._number in self.kill_on_serve or (
            self.kill_repo is not None and self._current == self.kill_repo
        )
        if killed:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def harness_voiding(units: set[str], side: str = "challenger"):
    """The fake benchmark, reporting `units` of `side` void for the harness's cause once their
    policy was driven: a simulator that lost the scene, which is nobody's loss."""
    import icil_fake_benchmark

    class HarnessVoiding(icil_fake_benchmark.FakeBenchmark):
        def run_command(self, *, unit, prompt, out_dir, policy_address, authkey_env, **extra):
            if unit["unit_id"] in units and Path(out_dir).parent.name == side:
                unit = {**unit, "fake_behaviour": "policy_then_void", "fake_void_cause": "harness"}
            return super().run_command(
                unit=unit,
                prompt=prompt,
                out_dir=out_dir,
                policy_address=policy_address,
                authkey_env=authkey_env,
                **extra,
            )

    return HarnessVoiding()


@dataclass
class InspectingFakeDocker(FakeDocker):
    """`FakeDocker`, answering the `docker inspect` the duel's adapter asks beyond `state`."""

    #: Containers the kernel killed for their memory limit.
    oom_killed: set[str] = field(default_factory=set)

    def _run(self, args, *, input_text=None, extra_env=None, timeout_s=None, check=True):
        args = list(args)
        names = {run[run.index("--name") + 1] for run in self.runs}
        if args[0] == "ps":
            wanted = {
                args[i + 1].removeprefix("label=") for i, a in enumerate(args) if a == "--filter"
            }
            listed = []
            for run in self.runs:
                name = run[run.index("--name") + 1]
                labels = {run[i + 1] for i, a in enumerate(run) if a == "--label"}
                if name not in self.removed and wanted <= labels:
                    listed.append(name)
            return subprocess.CompletedProcess(args, 0, "\n".join(listed) + "\n", "")
        if args[0] == "inspect" and "{{.State.OOMKilled}}" in args:
            name = args[-1]
            found = name in names and name not in self.removed
            out = ("true" if name in self.oom_killed else "false") if found else ""
            return subprocess.CompletedProcess(args, 0 if found else 1, out + "\n", "")
        raise NotImplementedError(f"the fake does not answer docker {' '.join(args[:2])}")


class RecordingReporter(LiveReporter):
    """Every frame a duel builds, kept rather than posted."""

    def __init__(self, spec: Any) -> None:
        super().__init__(spec, None, None)
        self.frames: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return True

    def post(self, frame: dict[str, Any], *, force: bool = False) -> bool:
        self.frames.append(frame)
        return True

    @property
    def phases(self) -> list[str]:
        """The phases in the order they were first posted."""
        return list(dict.fromkeys(f["phase"] for f in self.frames))
