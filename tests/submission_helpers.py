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

from icil_orchestrator.submissions.docker import (
    BuildFailed,
    BuildTimedOut,
    ContainerInfo,
    ContainerState,
)
from icil_orchestrator.submissions.image import INDEX_PROBE_IMAGE

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
class FakeRef:
    name: str
    ref: str
    target_commit: str


@dataclass
class FakeRefs:
    branches: list[FakeRef]
    tags: list[FakeRef]
    pull_requests: list[FakeRef] | None


@dataclass
class FakeCommit:
    commit_id: str


@dataclass
class FakeHub:
    """`HfApi.repo_info` over a table of repositories: `{repo: {revision: (sha, files)}}`, where
    `files` maps a path to its size. Every commit sha is also a revision of its repository.

    `list_repo_refs` and `list_repo_commits` answer from what `add` was told: each name is a branch
    at the commit, each of `tags` a tag, and `pr=N` makes it `refs/pr/N`'s alone. A commit added
    with none of them is the tip of a branch of its own, so it can be queued. `parent` puts the
    commit's history behind it: a commit added under a branch's name again moves the branch on and
    keeps the one before in its history."""

    repos: dict[str, dict[str, tuple[str, dict[str, int | None]]]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    trees: dict[str, Path] = field(default_factory=dict)
    #: `{repo: {name: sha}}` for branches and tags, `{repo: {number: sha}}` for pull requests.
    branches: dict[str, dict[str, str]] = field(default_factory=dict)
    tags: dict[str, dict[str, str]] = field(default_factory=dict)
    pull_requests: dict[str, dict[int, str]] = field(default_factory=dict)
    #: `{repo: {sha: [sha, parent, grandparent, ...]}}`, as the Hub lists a revision's commits.
    history: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    ref_calls: list[str] = field(default_factory=list)
    commit_calls: list[tuple[str, str]] = field(default_factory=list)

    def add(
        self,
        repo: str,
        sha: str,
        files: dict[str, int | None],
        *names: str,
        tags: tuple[str, ...] = (),
        pr: int | None = None,
        parent: str | None = None,
    ) -> None:
        entry = self.repos.setdefault(repo, {})
        entry[sha] = (sha, files)
        history = self.history.setdefault(repo, {})
        history[sha] = [sha, *history.get(parent, [parent])] if parent else [sha]
        if pr is not None:
            assert not names and not tags, "a pull request's commit is on no branch or tag"
            entry[f"refs/pr/{pr}"] = (sha, files)
            self.pull_requests.setdefault(repo, {})[pr] = sha
            return
        for name in (*names, *tags):
            entry[name] = (sha, files)
        branches = self.branches.setdefault(repo, {})
        for name in names or (() if tags else (f"unnamed-{sha[:12]}",)):
            branches[name] = sha
        for name in tags:
            self.tags.setdefault(repo, {})[name] = sha

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

    def list_repo_refs(self, repo_id, *, repo_type=None, include_pull_requests=False):
        self.ref_calls.append(repo_id)
        assert repo_type == "model"
        if repo_id not in self.repos:
            raise not_found(
                RepositoryNotFoundError, "Repository", "Repository not found", repo_id, "refs"
            )

        def refs(table: dict, prefix: str) -> list[FakeRef]:
            return [FakeRef(str(n), f"{prefix}{n}", s) for n, s in table.get(repo_id, {}).items()]

        return FakeRefs(
            branches=refs(self.branches, "refs/heads/"),
            tags=refs(self.tags, "refs/tags/"),
            pull_requests=refs(self.pull_requests, "refs/pr/") if include_pull_requests else None,
        )

    def list_repo_commits(self, repo_id, *, repo_type=None, revision=None):
        self.commit_calls.append((repo_id, revision))
        assert repo_type == "model"
        if repo_id not in self.repos:
            raise not_found(
                RepositoryNotFoundError, "Repository", "Repository not found", repo_id, revision
            )
        history = self.history.get(repo_id, {})
        if revision not in history:
            raise not_found(
                RevisionNotFoundError, "Revision", f"Invalid rev id: {revision}", repo_id, revision
            )
        return [FakeCommit(sha) for sha in history[revision]]


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


def pip_unreachable_log(requirement: str) -> str:
    """What pip prints when it cannot reach its index - and what any `setup.py` can print."""
    return (
        "WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None))"
        " after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection."
        "HTTPSConnection object at 0x7f>: Failed to establish a new connection: [Errno -3] "
        f"Temporary failure in name resolution')': /simple/{requirement}/\n"
        "ERROR: Could not find a version that satisfies the requirement "
        f"{requirement} (from versions: none)\n"
        f"ERROR: No matching distribution found for {requirement}"
    )


@dataclass(frozen=True)
class ProbeBuild:
    """One index probe as `FakeDocker.build` saw it, its context's entries listed at the time."""

    context: Path
    contents: list[str]
    dockerfile: str
    tag: str
    no_cache: bool
    timeout_s: float | None


@dataclass
class FakeDocker:
    """`Docker` without Docker: builds are recorded and given an id, and `run` starts
    `python -m icil_policy.serve` on the host, with the image's checkout in place of /submission
    and the mounted directory in place of /run/icil, so the start-and-hello path runs for real.
    An index probe (a build tagged `INDEX_PROBE_IMAGE`) is kept apart in `probes`, and reaches
    the index while `index_reachable` says so."""

    images: dict[str, str] = field(default_factory=dict)
    contexts: dict[str, Path] = field(default_factory=dict)
    builds: list[tuple[Path, str, str, dict]] = field(default_factory=list)
    tags: list[tuple[str, str]] = field(default_factory=list)
    runs: list[list[str]] = field(default_factory=list)
    run_envs: list[dict] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    processes: dict[str, subprocess.Popen] = field(default_factory=dict)
    #: The image each container was made from, by name, until it is removed.
    container_images: dict[str, str] = field(default_factory=dict)
    #: Every container's labels, by name, until it is removed; a test may add one it never ran.
    labels: dict[str, dict[str, str]] = field(default_factory=dict)
    #: When set, every build fails with this as its log.
    build_failure: str | None = None
    #: How long a build "takes": one over its timeout times out instead of building.
    build_seconds: float = 0.0
    #: The timeout each build was given.
    build_timeouts: list[float | None] = field(default_factory=list)
    #: Every index probe, in order; not in `builds`, and `build_failure` does not touch them.
    probes: list[ProbeBuild] = field(default_factory=list)
    #: Whether an index probe gets through; when not, it fails with pip's words for it.
    index_reachable: bool = True

    # -- images
    def image_id(self, ref):
        return self.images.get(ref)

    def find_image(self, image_id):
        return next((ref for ref, found in self.images.items() if found == image_id), None)

    def build(self, context, dockerfile, *, tag, build_args=None, timeout_s=None, no_cache=False):
        if tag.startswith(f"{INDEX_PROBE_IMAGE}:"):
            contents = sorted(os.listdir(context))
            self.probes.append(
                ProbeBuild(Path(context), contents, dockerfile, tag, no_cache, timeout_s)
            )
            if not self.index_reachable:
                raise BuildFailed(tag, pip_unreachable_log("pip"))
            self.images[tag] = (
                "sha256:" + hashlib.sha256(f"{tag}\n{dockerfile}".encode()).hexdigest()
            )
            return self.images[tag]
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

    def image_refs(self, repository):
        return [ref for ref in self.images if ref.startswith(f"{repository}:")]

    def remove_image(self, ref, *, force=True):
        users = [name for name, image in self.container_images.items() if image == ref]
        if users and not force:
            return f"conflict: unable to remove {ref}: container {users[0]} is using it"
        self.images.pop(ref, None)
        return ""

    # -- containers
    def run(self, args, *, env=None):
        args = list(args)
        self.runs.append(args)
        self.run_envs.append(dict(env or {}))
        name = args[args.index("--name") + 1]
        self.labels[name] = dict(
            args[i + 1].partition("=")[::2] for i, a in enumerate(args) if a == "--label"
        )
        mount = next(a for a in args if a.startswith("type=bind,src="))
        shared = mount.removeprefix("type=bind,src=").split(",")[0]
        # The authkey's `--env NAME` is the last flag; the image follows it, then the command.
        last_env = len(args) - 1 - args[::-1].index("--env")
        image = args[last_env + 2]
        self.container_images[name] = image
        checkout = self.contexts[self.images[image]]
        command = args[last_env + 3 :]
        # The shell that makes the home on the container's tmpfs has no tmpfs to make it on here:
        # the server runs directly.
        command = command[command.index("python") :]
        argv = [
            a.replace("/submission", str(checkout)).replace("/run/icil", shared) for a in command
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

    def policy_containers(self):
        return [
            ContainerInfo(name, self.state(name).running, dict(labels))
            for name, labels in self.labels.items()
            if labels.get("icil.orchestrator") == "policy"
        ]

    def logs(self, name, *, tail_lines=40):
        return ""

    def exec(self, name, argv, *, timeout_s=60):
        raise NotImplementedError("the fake has no inside to look at")

    def remove(self, name):
        self.removed.append(name)
        self.labels.pop(name, None)
        self.container_images.pop(name, None)
        process = self.processes.pop(name, None)
        if process is not None:
            process.kill()
            process.wait()

    def kill_all(self):
        for name in list(self.processes):
            self.remove(name)
