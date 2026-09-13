"""Stand-ins for the Hub and for Docker, so the submission path runs in the pure suite."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from icil_orchestrator.submissions.docker import BuildFailed, BuildTimedOut, ContainerState

SHA_A = "a" * 40
SHA_B = "b" * 40


def hub_response(status: int) -> httpx.Response:
    """A response as huggingface_hub's errors need one, with its request attached."""
    request = httpx.Request("GET", "https://huggingface.co/api/models/org/policy")
    return httpx.Response(status, request=request)


def not_found(kind, what: str, detail: str, repo: str, revision: str):
    """The 404 the Hub answers with, in the shape huggingface_hub raises it: a request id first,
    what was not found lines later, and the server's own words last and in `server_message`."""
    message = (
        "404 Client Error. (Request ID: Root=1-6aa6e45a-5ab94e211267ab2143004ac4)\n\n"
        f"{what} Not Found for url: https://huggingface.co/api/models/{repo}/revision/{revision}"
        f"?blobs=true.\n{detail}"
    )
    return kind(message, response=hub_response(404), server_message=detail)


@dataclass
class FakeSibling:
    rfilename: str
    size: int | None


@dataclass
class FakeInfo:
    sha: str
    siblings: list[FakeSibling]


@dataclass
class FakeHub:
    """`HfApi.repo_info` over a table of repositories: `{repo: {revision: (sha, files)}}`, where
    `files` maps a path to its size. Every commit sha is also a revision of its repository."""

    repos: dict[str, dict[str, tuple[str, dict[str, int | None]]]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    trees: dict[str, Path] = field(default_factory=dict)

    def add(self, repo: str, sha: str, files: dict[str, int | None], *names: str) -> None:
        entry = self.repos.setdefault(repo, {})
        entry[sha] = (sha, files)
        for name in names:
            entry[name] = (sha, files)

    def repo_info(self, repo_id, *, revision=None, repo_type=None, files_metadata=False):
        self.calls.append((repo_id, revision))
        assert repo_type == "model" and files_metadata
        if repo_id not in self.repos:
            raise not_found(
                RepositoryNotFoundError, "Repository", "Repository not found", repo_id, revision
            )
        if revision not in self.repos[repo_id]:
            raise not_found(
                RevisionNotFoundError, "Revision", f"Invalid rev id: {revision}", repo_id, revision
            )
        sha, files = self.repos[repo_id][revision]
        return FakeInfo(sha, [FakeSibling(p, s) for p, s in files.items()])


def write_policy_repo(root: Path, policy: str = "pkg.policy:Policy", **manifest: Any) -> Path:
    """A minimal competitor repository at `root`: a manifest and the package it names."""
    root.mkdir(parents=True, exist_ok=True)
    module = policy.partition(":")[0].split(".")
    package = root.joinpath(*module[:-1])
    package.mkdir(parents=True, exist_ok=True)
    for parent in [package, *package.parents]:
        if parent == root:
            break
        (parent / "__init__.py").touch()
    (package / f"{module[-1]}.py").write_text(
        "class Policy:\n"
        "    action_type = 'qpos'\n"
        "    def reset(self, seed): pass\n"
        "    def set_demonstration(self, arrays, info): pass\n"
        "    def act(self, observation): return {'action': [0.0]}\n"
    )
    lines = ["api: 1", f"policy: {policy}"]
    for key, value in manifest.items():
        lines.append(f"{key}: {value}")
    (root / "icil.yaml").write_text("\n".join(lines) + "\n")
    return root


# -- Docker, stood in for ----------------------------------------------------------------------

FAKE_BASE_DIGEST = "sha256:" + "b" * 64


@dataclass
class FakeDocker:
    """`Docker` without Docker: builds are recorded and given an id, and `run` starts
    `python -m icil_policy.serve` on the host, with the image's checkout in place of /submission
    and the mounted directory in place of /run/icil, so the start-and-hello path runs for real."""

    images: dict[str, str] = field(default_factory=dict)
    contexts: dict[str, Path] = field(default_factory=dict)
    builds: list[tuple[Path, str, str, dict]] = field(default_factory=list)
    tags: list[tuple[str, str]] = field(default_factory=list)
    runs: list[list[str]] = field(default_factory=list)
    run_envs: list[dict] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    processes: dict[str, subprocess.Popen] = field(default_factory=dict)
    #: When set, every build fails with this as its log.
    build_failure: str | None = None
    #: How long a build "takes": one over its timeout times out instead of building.
    build_seconds: float = 0.0
    #: The timeout each build was given.
    build_timeouts: list[float | None] = field(default_factory=list)

    # -- images
    def image_id(self, ref):
        return self.images.get(ref)

    def find_image(self, image_id):
        return next((ref for ref, found in self.images.items() if found == image_id), None)

    def build(self, context, dockerfile, *, tag, build_args=None, timeout_s=None):
        self.builds.append((Path(context), dockerfile, tag, dict(build_args or {})))
        self.build_timeouts.append(timeout_s)
        if self.build_failure is not None:
            raise BuildFailed(tag, self.build_failure)
        if timeout_s is not None and self.build_seconds > timeout_s:
            raise BuildTimedOut(tag, timeout_s)
        image_id = "sha256:" + hashlib.sha256(f"{context}\n{dockerfile}".encode()).hexdigest()
        self.images[tag] = image_id
        self.contexts[image_id] = Path(context)
        return image_id

    def tag(self, image, ref):
        self.images[ref] = image if image.startswith("sha256:") else self.images[image]
        self.tags.append((image, ref))

    def remove_image(self, ref):
        self.images.pop(ref, None)

    # -- containers
    def run(self, args, *, env=None):
        args = list(args)
        self.runs.append(args)
        self.run_envs.append(dict(env or {}))
        name = args[args.index("--name") + 1]
        mount = next(a for a in args if a.startswith("type=bind,src="))
        shared = mount.removeprefix("type=bind,src=").split(",")[0]
        image = args[args.index("--env") + 2]
        checkout = self.contexts[self.images[image]]
        argv = [
            a.replace("/submission", str(checkout)).replace("/run/icil", shared)
            for a in args[args.index(image) + 1 :]
        ]
        argv[0] = sys.executable
        environ = {"PATH": os.environ.get("PATH", ""), **(env or {})}
        self.processes[name] = subprocess.Popen(
            argv,
            env=environ,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=shared,
        )
        return "0123456789ab"

    def state(self, name):
        process = self.processes.get(name)
        if process is None:
            return ContainerState(False, None, "no such container")
        code = process.poll()
        return ContainerState(code is None, code)

    def logs(self, name, *, tail_lines=40):
        return ""

    def exec(self, name, argv, *, timeout_s=60):
        raise NotImplementedError("the fake has no inside to look at")

    def remove(self, name):
        self.removed.append(name)
        process = self.processes.pop(name, None)
        if process is not None:
            process.kill()
            process.wait()

    def kill_all(self):
        for name in list(self.processes):
            self.remove(name)
