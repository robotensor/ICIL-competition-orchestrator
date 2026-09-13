"""Stand-ins for the Hub and for Docker, so the submission path runs in the pure suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

SHA_A = "a" * 40
SHA_B = "b" * 40


def hub_response(status: int) -> httpx.Response:
    """A response as huggingface_hub's errors need one, with its request attached."""
    request = httpx.Request("GET", "https://huggingface.co/api/models/org/policy")
    return httpx.Response(status, request=request)


def not_found(kind, what: str):
    """The 404 the Hub answers with, as huggingface_hub raises it."""
    return kind(f"404 Client Error. {what} Not Found", response=hub_response(404))


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
            raise not_found(RepositoryNotFoundError, "Repository")
        if revision not in self.repos[repo_id]:
            raise not_found(RevisionNotFoundError, "Revision")
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
