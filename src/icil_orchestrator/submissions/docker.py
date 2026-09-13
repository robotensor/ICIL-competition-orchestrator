"""The `docker` command line, called with an allow-listed environment.

Every image build and container run goes through this one class, so the tests can stand a fake
in for it and so the rule holds in one place: the docker client gets the same environment a
benchmark subprocess gets (`benchmark_environment`) plus its own `DOCKER_*` variables, and the
policy's authkey only for `run`, by variable name. Never `HF_TOKEN`, never the live token: nothing
the orchestrator holds reaches a container unless it is passed here on purpose.
"""

from __future__ import annotations

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


class BuildFailed(DockerError):
    """`docker build` ran and failed; `log` is the tail of its output."""

    def __init__(self, tag: str, log: str) -> None:
        self.tag = tag
        self.log = log
        super().__init__(f"building {tag} failed:\n{log}")


@dataclass(frozen=True)
class ContainerState:
    running: bool
    exit_code: int | None
    error: str = ""


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
            raise DockerError(f"docker {args[0]} did not finish within {timeout_s:g}s") from None
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
    ) -> str:
        """Build `dockerfile` (its text; read from stdin) with `context`, tagged `tag`; the image
        id. `BuildFailed` with the log's tail when the build itself fails."""
        args = ["build", "--tag", tag, "--file", "-"]
        for key, value in (build_args or {}).items():
            args += ["--build-arg", f"{key}={value}"]
        with tempfile.TemporaryDirectory(prefix="icil-build-") as tmp:
            iidfile = Path(tmp) / "iid"
            args += ["--iidfile", str(iidfile), str(context)]
            done = self._run(args, input_text=dockerfile, timeout_s=timeout_s, check=False)
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

    def remove_image(self, ref: str) -> None:
        self._run(["rmi", "--force", ref], check=False, timeout_s=300)

    # -- containers -------------------------------------------------------------------------

    def run(self, args: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
        """`docker run <args>`: the container id. `env` reaches the docker client, not the
        container - a `--env NAME` among `args` is what carries a value in."""
        done = self._run(["run", *args], extra_env=env, timeout_s=600)
        return done.stdout.strip()

    def state(self, name: str) -> ContainerState:
        done = self._run(
            [
                "inspect",
                "--type",
                "container",
                "--format",
                "{{.State.Running}} {{.State.ExitCode}} {{.State.Error}}",
                name,
            ],
            check=False,
            timeout_s=60,
        )
        if done.returncode != 0:
            return ContainerState(False, None, f"no such container: {_tail(done.stderr, 200)}")
        running, _, rest = done.stdout.strip().partition(" ")
        code, _, error = rest.partition(" ")
        return ContainerState(running == "true", int(code) if code.isdigit() else None, error)

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
