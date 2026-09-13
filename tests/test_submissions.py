"""Resolving, fetching and checking a submission, with the Hub stood in for."""

from __future__ import annotations

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from icil_orchestrator.ids import SubmissionRef
from icil_orchestrator.submissions import SubmissionError, SubmissionRejected
from icil_orchestrator.submissions.resolve import Resolved, resolve
from submission_helpers import SHA_A, SHA_B, FakeHub, hub_response

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
    with pytest.raises(SubmissionRejected, match="Repository Not Found") as info:
        resolve("org/missing", "main", api=hub)
    assert info.value.step == "resolve"
    with pytest.raises(SubmissionRejected, match="Revision Not Found"):
        resolve("org/policy", "no-such-branch", api=hub)
    with pytest.raises(SubmissionRejected, match="Revision Not Found"):
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


def test_an_entry_the_hub_gives_no_size_for_counts_nothing_until_it_is_fetched(hub):
    hub.add("org/lfs", SHA_A, {"icil.yaml": 80, "weights.pt": None}, "main")
    resolved = resolve("org/lfs", "main", api=hub)
    assert resolved.declared_bytes == 80
    assert [f.size for f in resolved.files] == [80, None]
