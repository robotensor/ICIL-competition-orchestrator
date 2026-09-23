"""Resolving, fetching and checking a submission, with the Hub stood in for."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from submission_helpers import SHA_A, SHA_B, FakeHub, hub_response, write_policy_repo
from vector_orchestrator.ids import SubmissionRef, is_commit_sha
from vector_orchestrator.spec import load_spec_file
from vector_orchestrator.submissions import SubmissionError, SubmissionRejected
from vector_orchestrator.submissions.checks import check_repository, regular_file_inside
from vector_orchestrator.submissions.fetch import (
    HubFetcher,
    LocalFetcher,
    RepoCache,
    measure,
    tree_hash,
)
from vector_orchestrator.submissions.resolve import (
    Resolved,
    check_reachable,
    resolve,
    resolve_for_queue,
)

# -- resolve ------------------------------------------------------------------------------------


@pytest.fixture
def hub():
    hub = FakeHub()
    hub.add("org/policy", SHA_A, {"icil.yaml": 80, "policy.py": 1200, "weights.pt": 4000}, "main")
    hub.add("org/policy", SHA_B, {"icil.yaml": 80, "policy.py": 1300}, "v2", "dev")
    return hub


def test_a_branch_or_tag_resolves_to_its_commit_and_a_sha_to_itself(hub):
    assert resolve("org/policy", "main", api=hub).sha == SHA_A
    assert resolve("org/policy", "v2", api=hub).sha == SHA_B
    assert resolve("org/policy", "dev", api=hub).sha == SHA_B
    assert resolve("org/policy", SHA_A, api=hub).sha == SHA_A
    # The Hub was asked exactly once each: resolution happens at queue time and never again.
    assert hub.calls == [("org/policy", r) for r in ("main", "v2", "dev", SHA_A)]


def test_the_resolved_ref_is_keyed_by_the_sha_and_lists_the_files_with_sizes(hub):
    resolved = resolve("org/policy", "main", api=hub)
    assert resolved == Resolved(repo="org/policy", revision="main", sha=SHA_A, files=resolved.files)
    assert resolved.ref == SubmissionRef.resolved("org/policy", SHA_A)
    assert resolved.ref.key == SubmissionRef.make("org/policy", SHA_A).key
    assert {f.path: f.size for f in resolved.files} == {
        "icil.yaml": 80,
        "policy.py": 1200,
        "weights.pt": 4000,
    }
    assert resolved.declared_bytes == 5280


def test_what_the_hub_does_not_have_is_rejected_with_the_reason(hub):
    """The reason carries the Hub's own words, which its 404 puts lines after the request id."""
    with pytest.raises(SubmissionRejected) as info:
        resolve("org/missing", "main", api=hub)
    assert info.value.step == "resolve"
    assert info.value.reason == "org/missing@main: repository not found"
    with pytest.raises(SubmissionRejected) as info:
        resolve("org/policy", "no-such-branch", api=hub)
    assert info.value.reason == (
        "org/policy@no-such-branch: revision not found: Invalid rev id: no-such-branch"
    )
    assert "Request ID" not in info.value.reason
    with pytest.raises(SubmissionRejected, match="revision not found: Invalid rev id: c+$"):
        resolve("org/policy", "c" * 40, api=hub)
    with pytest.raises(SubmissionRejected, match="is not a Hugging Face repo id"):
        resolve("not a repo", "main", api=hub)
    with pytest.raises(SubmissionRejected, match="no revision"):
        resolve("org/policy", "", api=hub)
    assert ("not a repo", "main") not in hub.calls


def test_a_hub_that_cannot_be_asked_is_the_harness_problem_not_the_submissions(hub):
    class Down:
        def repo_info(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    class Overloaded:
        def repo_info(self, *args, **kwargs):
            raise HfHubHTTPError("503 Server Error", response=hub_response(503))

    class Lying:
        def repo_info(self, *args, **kwargs):
            return type("Info", (), {"sha": "not-a-sha", "siblings": []})()

    class Swapping:
        def repo_info(self, *args, **kwargs):
            return type("Info", (), {"sha": SHA_B, "siblings": []})()

    with pytest.raises(SubmissionError, match="unreachable"):
        resolve("org/policy", "main", api=Down())
    with pytest.raises(SubmissionError, match="503"):
        resolve("org/policy", "main", api=Overloaded())
    with pytest.raises(SubmissionError, match="no commit sha"):
        resolve("org/policy", "main", api=Lying())
    with pytest.raises(SubmissionError, match="to another"):
        resolve("org/policy", SHA_A, api=Swapping())


OLD, TIP, TAGGED, PROPOSED, CLOSED = (c * 40 for c in "12345")


@pytest.fixture
def history():
    """org/policy: main moved from OLD to TIP, v1 tags TAGGED off OLD, pull request 1 proposes
    PROPOSED on TIP, and pull request 2, closed, left CLOSED on no ref at all."""
    hub = FakeHub()
    files = {"icil.yaml": 80}
    hub.add("org/policy", OLD, files, "main")
    hub.add("org/policy", TIP, files, "main", parent=OLD)
    hub.add("org/policy", TAGGED, files, tags=("v1",), parent=OLD)
    hub.add("org/policy", PROPOSED, files, pr=1, parent=TIP)
    hub.add("org/policy", CLOSED, files, pr=2, parent=OLD)
    del hub.pull_requests["org/policy"][2]
    return hub


def test_a_commit_is_queued_only_from_a_branch_or_a_tag(history):
    """The Hub serves any commit it holds by its sha, a pull request's included, and anyone on the
    Hub can open a pull request: a commit only a pull request holds is refused for the queue."""
    for sha in (TIP, OLD, TAGGED):
        assert resolve_for_queue("org/policy", sha, api=history).sha == sha
    assert resolve_for_queue("org/policy", "main", api=history).sha == TIP
    assert resolve_for_queue("org/policy", "v1", api=history).sha == TAGGED
    for sha in (PROPOSED, CLOSED):
        with pytest.raises(SubmissionRejected) as info:
            resolve_for_queue("org/policy", sha, api=history)
        assert info.value.step == "resolve"
        assert info.value.reason == (
            f"org/policy@{sha}: no branch or tag of the repository holds this commit; a pull "
            "request's commit (refs/pr/N), or one no branch holds any more, is not queued"
        )
    # Resolving alone still serves it: a king already crowned is not refused for where it lives.
    assert resolve("org/policy", PROPOSED, api=history).sha == PROPOSED

    history.calls.clear()
    history.ref_calls.clear()
    for name in ("refs/pr/1", "refs/heads/main"):
        with pytest.raises(
            SubmissionRejected, match="a ref under refs/, a pull request's included"
        ):
            resolve_for_queue("org/policy", name, api=history)
    assert history.calls == [] and history.ref_calls == [], "the Hub was asked about a refs/ name"


def test_a_history_is_asked_for_only_when_no_tip_is_the_commit(history):
    check_reachable("org/policy", TIP, api=history)
    check_reachable("org/policy", TAGGED, api=history)
    assert history.commit_calls == [], "a branch's or tag's tip needs no history"
    check_reachable("org/policy", OLD, api=history)
    assert history.commit_calls == [("org/policy", TIP)], "the walk went past the first holder"


def test_a_hub_that_cannot_confirm_a_commit_in_time_is_the_harness_problem(history):
    with pytest.raises(SubmissionError, match="took too long listing org/policy's history"):
        check_reachable("org/policy", PROPOSED, api=history, deadline=time.monotonic() - 1)

    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("[Errno 101] Network is unreachable")

    history.list_repo_refs = unreachable
    with pytest.raises(SubmissionError, match="the Hub is unreachable resolving org/policy@"):
        resolve_for_queue("org/policy", "main", api=history)
    with pytest.raises(SubmissionRejected, match="repository not found"):
        check_reachable("org/missing", TIP, api=FakeHub())


def test_an_entry_the_hub_gives_no_size_for_counts_nothing_and_is_not_downloaded(
    hub, tmp_path, source
):
    """With `files_metadata` the Hub sizes every file, LFS ones included; an entry it does not
    size cannot be held to max_repo_bytes before it is on disk, so it is not fetched at all -
    the Hub's answer is the harness's problem, not the entry's."""
    hub.add("org/lfs", SHA_A, {"icil.yaml": 80, "weights.pt": None}, "main")
    resolved = resolve("org/lfs", "main", api=hub)
    assert resolved.declared_bytes == 80, "a lower bound: what was sized"
    assert [f.size for f in resolved.files] == [80, None]
    calls: list = []
    fetcher = HubFetcher(
        RepoCache(tmp_path / "cache", 10**9), api=hub, download=fake_download(source, calls)
    )
    with pytest.raises(SubmissionError, match="declared no size for 1 file.*weights.pt") as info:
        fetcher.fetch(resolved)
    assert not isinstance(info.value, SubmissionRejected)
    assert calls == [] and not (tmp_path / "cache" / SHA_A).exists()


# -- fetch --------------------------------------------------------------------------------------


def fake_download(source: Path, calls: list):
    """`snapshot_download` stood in for: copies `source` into `local_dir` and leaves the Hub's
    bookkeeping directory behind, as the real one does."""

    def download(repo_id, *, revision, repo_type, local_dir):
        calls.append((repo_id, revision, repo_type))
        shutil.copytree(source, local_dir, dirs_exist_ok=True)
        bookkeeping = Path(local_dir) / ".cache" / "huggingface" / "download"
        bookkeeping.mkdir(parents=True)
        (bookkeeping / "policy.py.metadata").write_text("etag")

    return download


@pytest.fixture
def source(tmp_path):
    root = write_policy_repo(tmp_path / "source")
    (root / "weights.bin").write_bytes(b"\0" * 4000)
    return root


def test_fetch_downloads_once_into_a_checkout_addressed_by_the_sha(spec, tmp_path, hub, source):
    calls: list = []
    cache = RepoCache(tmp_path / "cache", spec.submission["max_repo_bytes"])
    fetcher = HubFetcher(cache, api=hub, download=fake_download(source, calls))
    resolved = fetcher.resolve("org/policy", "main")

    fetched = fetcher.fetch(resolved)
    assert fetched.root == tmp_path / "cache" / SHA_A / "repo"
    assert not fetched.cached and calls == [("org/policy", SHA_A, "model")]
    assert sorted(p.name for p in fetched.root.iterdir()) == ["icil.yaml", "pkg", "weights.bin"]
    assert not (fetched.root / ".cache").exists(), "the Hub's bookkeeping is not the repository"
    assert (fetched.bytes, fetched.files) == measure(source)
    marker = json.loads((tmp_path / "cache" / SHA_A / "fetched.json").read_text())
    assert (marker["repo"], marker["sha"], marker["bytes"]) == ("org/policy", SHA_A, fetched.bytes)

    again = fetcher.fetch(resolved)
    assert again.cached and again.root == fetched.root and len(calls) == 1
    # Another revision name for the same commit is the same checkout.
    hub.add("org/policy", SHA_A, {}, "release")
    assert fetcher.fetch(fetcher.resolve("org/policy", "release")).cached and len(calls) == 1


def test_fetch_refuses_a_repository_over_max_repo_bytes_before_and_after_download(
    tmp_path, hub, source
):
    calls: list = []
    # The Hub declares 5280 bytes for org/policy@main; a limit under that refuses it unfetched.
    small = HubFetcher(
        RepoCache(tmp_path / "small", 5000), api=hub, download=fake_download(source, calls)
    )
    with pytest.raises(
        SubmissionRejected, match="declares 5280 bytes, over max_repo_bytes"
    ) as info:
        small.fetch(small.resolve("org/policy", "main"))
    assert info.value.step == "fetch" and calls == []
    assert not (tmp_path / "small" / SHA_A).exists()

    # Declared sizes are what the Hub says; what lands on disk is measured again.
    hub.add("org/policy", SHA_B, {"icil.yaml": 1}, "tiny")
    lying = HubFetcher(
        RepoCache(tmp_path / "lying", 100), api=hub, download=fake_download(source, calls)
    )
    with pytest.raises(SubmissionRejected, match="bytes on disk, over max_repo_bytes"):
        lying.fetch(lying.resolve("org/policy", "tiny"))
    assert len(calls) == 1
    assert not (tmp_path / "lying" / SHA_B / "repo").exists()
    assert not (tmp_path / "lying" / SHA_B / "partial").exists()
    assert not (tmp_path / "lying" / SHA_B / "fetched.json").exists()


def test_a_download_that_fails_is_the_harness_problem_and_leaves_nothing_behind(
    spec, tmp_path, hub
):
    def broken(repo_id, **kwargs):
        Path(kwargs["local_dir"], "half.bin").write_bytes(b"x" * 10)
        raise HfHubHTTPError("502 Bad Gateway", response=hub_response(502))

    fetcher = HubFetcher(
        RepoCache(tmp_path / "cache", spec.submission["max_repo_bytes"]), api=hub, download=broken
    )
    with pytest.raises(SubmissionError, match="downloading org/policy@a+ failed: HfHubHTTPError"):
        fetcher.fetch(fetcher.resolve("org/policy", "main"))
    assert not (tmp_path / "cache" / SHA_A / "repo").exists()
    assert not (tmp_path / "cache" / SHA_A / "partial").exists()
    assert fetcher.cache.lookup(fetcher.resolve("org/policy", "main")) is None


def test_a_local_directory_is_addressed_by_its_tree_and_copied_links_as_links(
    spec, tmp_path, source
):
    (source / "pkg" / "__pycache__").mkdir()
    (source / "pkg" / "__pycache__" / "policy.cpython-310.pyc").write_bytes(b"\0")
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    os.symlink("weights.bin", source / "link.bin")
    os.symlink("/etc/hostname", source / "outside.txt")

    cache = RepoCache(tmp_path / "cache", spec.submission["max_repo_bytes"])
    fetcher = LocalFetcher(cache, source)
    resolved = fetcher.resolve("local/policy", "main")
    assert is_commit_sha(resolved.sha) and resolved.revision == "main"
    assert resolved.ref == SubmissionRef.resolved("local/policy", resolved.sha)
    assert {f.path for f in resolved.files} == {
        "icil.yaml",
        "pkg/__init__.py",
        "pkg/policy.py",
        "weights.bin",
        "link.bin",
        "outside.txt",
    }, "neither .git nor __pycache__ is part of what would be pushed"

    fetched = fetcher.fetch(resolved)
    assert fetched.root == tmp_path / "cache" / resolved.sha / "repo"
    assert not (fetched.root / ".git").exists() and not (fetched.root / "pkg/__pycache__").exists()
    assert os.readlink(fetched.root / "link.bin") == "weights.bin"
    assert os.readlink(fetched.root / "outside.txt") == "/etc/hostname"
    assert fetched.bytes == sum(f.size for f in resolved.files if f.size is not None) < 5000, (
        "a link counts for itself, not for what it points at"
    )
    assert fetcher.fetch(fetcher.resolve("local/policy", "main")).cached
    # The directory stands in for the Hub, not for the shape of a ref.
    with pytest.raises(SubmissionRejected, match="is not a Hugging Face repo id") as info:
        fetcher.resolve("not a repo id", "main")
    assert info.value.step == "resolve"
    with pytest.raises(SubmissionRejected, match="no revision"):
        fetcher.resolve("local/policy", "")

    # Same tree, same address; a changed byte is another submission.
    copy = shutil.copytree(source, tmp_path / "copy", symlinks=True)
    assert tree_hash(copy) == resolved.sha
    (copy / "pkg" / "policy.py").write_text("class Policy: pass\n")
    assert tree_hash(copy) != resolved.sha
    os.chmod(source / "pkg" / "policy.py", 0o755)
    assert tree_hash(source) != resolved.sha, "the executable bit is part of the tree, as in git"


# -- checks -------------------------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parents[1] / "packages" / "vector-policy" / "examples"


def test_the_replay_example_passes_the_manifest_check(spec):
    manifest = check_repository(EXAMPLES / "replay_policy", spec)
    assert manifest.policy == "replay.policy:ReplayPolicy"
    assert manifest.requirements == "requirements.txt"
    assert manifest.api == spec.submission["manifest_api"]


def test_a_manifest_that_is_not_a_plain_file_of_the_repository_is_refused(spec, tmp_path):
    root = write_policy_repo(tmp_path / "repo")
    assert check_repository(root, spec).policy == "pkg.policy:Policy"

    (root / "icil.yaml").rename(root / "real.yaml")
    with pytest.raises(SubmissionRejected, match="icil.yaml.*is not in the repository") as info:
        check_repository(root, spec)
    assert info.value.step == "manifest"

    os.symlink("real.yaml", root / "icil.yaml")
    with pytest.raises(SubmissionRejected, match="goes through a symbolic link"):
        check_repository(root, spec)
    (root / "icil.yaml").unlink()

    (root / "icil.yaml").mkdir()
    with pytest.raises(SubmissionRejected, match="is not a regular file"):
        check_repository(root, spec)
    (root / "icil.yaml").rmdir()

    os.mkfifo(root / "icil.yaml")
    with pytest.raises(SubmissionRejected, match="is not a regular file"):
        check_repository(root, spec)  # and did not block opening the pipe


def test_a_manifests_problems_are_the_rejections_reason(spec, tmp_path):
    root = write_policy_repo(tmp_path / "repo")
    (root / "icil.yaml").write_text("api: 2\npolicy: not a class\nextra: 1\n")
    with pytest.raises(SubmissionRejected) as info:
        check_repository(root, spec)
    reason = info.value.reason
    assert "api: must be 1" in reason and "policy: must be module:Class" in reason
    assert "unknown key(s) 'extra'" in reason

    (root / "icil.yaml").write_text("api: 1\npolicy: pkg.policy:Policy\n")
    other = json.loads(spec.path.read_text())
    other["submission"]["manifest_api"] = 2
    (tmp_path / "spec.json").write_text(json.dumps(other))
    with pytest.raises(SubmissionRejected, match="api 1 is not this competition's 2"):
        check_repository(root, load_spec_file(tmp_path / "spec.json"))


def test_requirements_must_be_a_plain_file_inside_the_repository(spec, tmp_path):
    root = write_policy_repo(tmp_path / "repo", requirements="requirements.txt")
    (root / "requirements.txt").write_text("numpy\n")
    assert check_repository(root, spec).requirements == "requirements.txt"

    (tmp_path / "outside.txt").write_text("evil\n")
    (root / "requirements.txt").unlink()
    os.symlink(tmp_path / "outside.txt", root / "requirements.txt")
    with pytest.raises(SubmissionRejected, match="requirements.*symbolic link|leaves the repo"):
        check_repository(root, spec)
    (root / "requirements.txt").unlink()

    write_policy_repo(root, requirements="../outside.txt")
    with pytest.raises(SubmissionRejected, match="leaves the repository"):
        check_repository(root, spec)

    # A link to a file that is inside: vector_policy resolves it happily; this check does not.
    (root / "deps").mkdir()
    (root / "deps" / "requirements.txt").write_text("numpy\n")
    os.symlink("deps/requirements.txt", root / "requirements.txt")
    write_policy_repo(root, requirements="requirements.txt")
    with pytest.raises(SubmissionRejected, match="goes through a symbolic link \\(requirements"):
        check_repository(root, spec)
    (root / "requirements.txt").unlink()

    os.symlink("deps", root / "linked")
    write_policy_repo(root, requirements="linked/requirements.txt")
    with pytest.raises(SubmissionRejected, match="goes through a symbolic link \\(linked\\)"):
        check_repository(root, spec)


def test_regular_file_inside_refuses_paths_that_leave_or_are_absolute(tmp_path):
    (tmp_path / "f").write_text("x")
    assert regular_file_inside(tmp_path, "f", what="it") == tmp_path / "f"
    assert regular_file_inside(tmp_path, "./f", what="it") == tmp_path / "f"
    for bad in ("/etc/passwd", "../f", "", "a/../f"):
        with pytest.raises(SubmissionRejected, match="is not a path inside the repository"):
            regular_file_inside(tmp_path, bad, what="it")
