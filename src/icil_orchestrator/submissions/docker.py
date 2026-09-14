"""The `docker` command line, called with an allow-listed environment.

Every image build and container run goes through this one class, so the tests can stand a fake
in for it and so the rule holds in one place: the docker client gets the same environment a
benchmark subprocess gets (`benchmark_environment`) plus its own `DOCKER_*` variables, and the
policy's authkey only for `run`, by variable name. Never `HF_TOKEN`, never the live token: nothing
the orchestrator holds reaches a container unless it is passed here on purpose.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..benchmarks.subprocess_runner import benchmark_environment
from .errors import SubmissionError

#: What the docker client itself reads, on top of `ENV_ALLOW`.
DOCKER_ENV = (
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
    "DOCKER_API_VERSION",
    "DOCKER_BUILDKIT",
)

#: How much of a build's or a container's output a failure quotes.
TAIL_CHARS = 2000

#: Every container the orchestrator starts carries it, so `docker ps --filter label=...` lists
#: exactly them and a crashed orchestrator's leftovers can be found.
CONTAINER_LABEL = "icil.orchestrator=policy"


class DockerError(SubmissionError):
    """Docker itself could not do what was asked: not installed, not running, refused."""


class DockerTimeout(DockerError):
    """A docker command did not finish within its timeout and was killed."""

    def __init__(self, command: str, timeout_s: float) -> None:
        self.command = command
        self.timeout_s = timeout_s
        super().__init__(f"docker {command} did not finish within {timeout_s:g}s")


class BuildFailed(DockerError):
    """`docker build` ran and failed; `log` is the tail of its output."""

    def __init__(self, tag: str, log: str) -> None:
        self.tag = tag
        self.log = log
        super().__init__(f"building {tag} failed:\n{log}")


class BuildTimedOut(DockerError):
    """`docker build` did not finish within `timeout_s`. The client was killed, and BuildKit
    cancels a build whose client is gone, so nothing of it keeps running in the daemon."""

    def __init__(self, tag: str, timeout_s: float) -> None:
        self.tag = tag
        self.timeout_s = timeout_s
        super().__init__(f"building {tag} did not finish within {timeout_s:g}s")


@dataclass(frozen=True)
class ContainerState:
    running: bool
    exit_code: int | None
    error: str = ""
    #: The kernel killed the container for its memory limit (`State.OOMKilled`).
    oom_killed: bool = False


@dataclass(frozen=True)
class ContainerInfo:
    """A container carrying `CONTAINER_LABEL`, as `Docker.policy_containers` lists it."""

    name: str
    running: bool
    labels: dict[str, str]


def docker_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """What the docker client runs with: the benchmark allow-list plus `DOCKER_*`."""
    return benchmark_environment(os.environ if source is None else source, "", keep=DOCKER_ENV)


class Docker:
    def __init__(self, binary: str = "docker", environ: Mapping[str, str] | None = None) -> None:
        self.binary = binary
        self.environ = docker_environment(environ)

    def _run(
        self,
        args: Sequence[str],
        *,
        input_text: str | None = None,
        extra_env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        env = dict(self.environ)
        if extra_env:
            env.update(extra_env)
        try:
            done = subprocess.run(
                [self.binary, *args],
                input=input_text,
                stdin=None if input_text is not None else subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except FileNotFoundError:
            raise DockerError(f"{self.binary} is not installed or not on PATH") from None
        except subprocess.TimeoutExpired:
            raise DockerTimeout(args[0], timeout_s or 0.0) from None
        if check and done.returncode != 0:
            raise DockerError(
                f"docker {' '.join(args[:2])} failed ({done.returncode}): {_tail(done.stderr)}"
            )
        return done

    # -- images -----------------------------------------------------------------------------

    def image_id(self, ref: str) -> str | None:
        """`sha256:<hex>` of the local image `ref` names, or None when there is none."""
        done = self._run(
            ["image", "inspect", "--format", "{{.Id}}", ref], check=False, timeout_s=60
        )
        return done.stdout.strip() if done.returncode == 0 else None

    def find_image(self, image_id: str) -> str | None:
        """Some local reference to the image with this id, or None."""
        done = self._run(
            ["images", "--no-trunc", "--format", "{{.ID}}\t{{.Repository}}:{{.Tag}}"],
            timeout_s=60,
        )
        for line in done.stdout.splitlines():
            found, _, ref = line.partition("\t")
            if found.strip() == image_id:
                return ref.strip()
        return None

    def build(
        self,
        context: Path,
        dockerfile: str,
        *,
        tag: str,
        build_args: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        no_cache: bool = False,
    ) -> str:
        """Build `dockerfile` (its text; read from stdin) with `context`, tagged `tag`; the image
        id. `BuildFailed` with the log's tail when the build itself fails, `BuildTimedOut` when
        it is not done within `timeout_s`. `no_cache` runs every step again, for a build whose
        point is what its steps do now rather than the image they leave."""
        args = ["build", "--tag", tag, "--file", "-"]
        if no_cache:
            args.append("--no-cache")
        for key, value in (build_args or {}).items():
            args += ["--build-arg", f"{key}={value}"]
        with tempfile.TemporaryDirectory(prefix="icil-build-") as tmp:
            iidfile = Path(tmp) / "iid"
            args += ["--iidfile", str(iidfile), str(context)]
            try:
                done = self._run(args, input_text=dockerfile, timeout_s=timeout_s, check=False)
            except DockerTimeout as exc:
                raise BuildTimedOut(tag, exc.timeout_s) from None
            if done.returncode != 0:
                raise BuildFailed(tag, _tail(done.stderr + done.stdout))
            try:
                image_id = iidfile.read_text().strip()
            except OSError:
                image_id = ""
        if not image_id:
            image_id = self.image_id(tag) or ""
        if not image_id:
            raise DockerError(f"docker build reported success but {tag} has no image id")
        return image_id

    def tag(self, image: str, ref: str) -> None:
        self._run(["tag", image, ref], timeout_s=60)

    def image_refs(self, repository: str) -> list[str]:
        """Every local `repository:tag` of `repository`; an untagged image has none."""
        done = self._run(
            ["images", "--format", "{{.Repository}}:{{.Tag}}", repository], timeout_s=60
        )
        refs = (line.strip() for line in done.stdout.splitlines())
        return [
            ref
            for ref in refs
            if ref.startswith(f"{repository}:") and ref != f"{repository}:<none>"
        ]

    def remove_image(self, ref: str, *, force: bool = True) -> str:
        """`docker rmi`: "" once `ref` is gone, docker's reason when it is not - without `force`,
        an image a container (running or not) was made from is refused."""
        done = self._run(["rmi", *(["--force"] if force else []), ref], check=False, timeout_s=300)
        return "" if done.returncode == 0 else _tail(done.stderr, 300) or f"exit {done.returncode}"

    # -- containers -------------------------------------------------------------------------

    def run(self, args: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
        """`docker run <args>`: the container id. `env` reaches the docker client, not the
        container - a `--env NAME` among `args` is what carries a value in."""
        done = self._run(["run", *args], extra_env=env, timeout_s=600)
        return done.stdout.strip()

    def state(self, name: str) -> ContainerState:
        """Whether `name` runs, its exit code, docker's error and whether the kernel killed it for
        its memory limit; a container docker does not know has no exit code."""
        done = self._run(
            [
                "inspect",
                "--type",
                "container",
                "--format",
                "{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}} {{.State.Error}}",
                name,
            ],
            check=False,
            timeout_s=60,
        )
        if done.returncode != 0:
            return ContainerState(False, None, f"no such container: {_tail(done.stderr, 200)}")
        running, _, rest = done.stdout.strip().partition(" ")
        oom_killed, _, rest = rest.partition(" ")
        code, _, error = rest.partition(" ")
        return ContainerState(
            running == "true",
            int(code) if code.isdigit() else None,
            error,
            oom_killed=oom_killed == "true",
        )

    def policy_containers(self) -> list[ContainerInfo]:
        """Every container carrying `CONTAINER_LABEL`, running or not, with its labels."""
        listed = self._run(
            ["ps", "--all", "--quiet", "--no-trunc", "--filter", f"label={CONTAINER_LABEL}"],
            timeout_s=60,
        ).stdout.split()
        if not listed:
            return []
        # One JSON object a line; a container removed since `ps` is an error line, skipped.
        template = (
            '{"name": {{json .Name}}, "running": {{json .State.Running}}, '
            '"labels": {{json .Config.Labels}}}'
        )
        done = self._run(
            ["inspect", "--type", "container", "--format", template, *listed],
            check=False,
            timeout_s=60,
        )
        found = []
        for line in done.stdout.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            labels = row.get("labels") or {}
            found.append(
                ContainerInfo(
                    name=str(row.get("name", "")).lstrip("/"),
                    running=row.get("running") is True,
                    labels={str(k): str(v) for k, v in labels.items()},
                )
            )
        return found

    def logs(self, name: str, *, tail_lines: int = 40) -> str:
        done = self._run(["logs", "--tail", str(tail_lines), name], check=False, timeout_s=60)
        return (done.stdout + done.stderr).strip()

    def exec(
        self, name: str, argv: Sequence[str], *, timeout_s: float = 60
    ) -> subprocess.CompletedProcess:
        """Run `argv` inside a running container, as its user; never raises on its exit status."""
        return self._run(["exec", name, *argv], check=False, timeout_s=timeout_s)

    def remove(self, name: str) -> None:
        """`docker rm -f`: gone, whatever state it was in. Raises nothing."""
        try:
            self._run(["rm", "--force", name], check=False, timeout_s=120)
        except DockerError:
            pass


def _tail(text: str, limit: int = TAIL_CHARS) -> str:
    return text.strip()[-limit:]
