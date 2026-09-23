"""Mirroring to a Hugging Face dataset, against a stand-in `HfApi`: no network in the pure tests."""

from __future__ import annotations

from types import SimpleNamespace

import huggingface_hub
import pytest

from vector_orchestrator.canon import Signer
from vector_orchestrator.store import mirror
from vector_orchestrator.store.writer import Store, store_lock


class FakeApi:
    remote: set[str] = set()
    commits: list[tuple[str, list]] = []

    def __init__(self, token=None):
        self.token = token

    def create_repo(self, repo, **kw):
        pass

    def repo_info(self, repo, **kw):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=f) for f in FakeApi.remote])

    def create_commit(self, *, repo_id, repo_type, operations, commit_message):
        FakeApi.commits.append((commit_message, operations))
        return SimpleNamespace(oid="c0ffee")


@pytest.fixture
def api(monkeypatch):
    FakeApi.remote, FakeApi.commits = set(), []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    return FakeApi


@pytest.fixture
def store(spec, tmp_path):
    s = Store(tmp_path / "store", spec, Signer.generate())
    s.init(s.signer.verify_key_hex)
    with store_lock(s.root):
        pass
    (s.root / "tracks" / "franka_1arm" / "queue.json.tmp").write_text("{")
    return s


def test_mirror_lists_only_the_store_files(store):
    # Everything a wrong root might hold beside a store, all of it published if it were uploaded.
    (store.root / "keys").mkdir()
    (store.root / "keys" / "orchestrator.ed25519").write_text("00" * 32 + "\n")
    (store.root / "notes.md").write_text("scratch")
    (store.root / "events" / "franka_1arm").mkdir(parents=True)
    (store.root / "events" / "franka_1arm" / "abc123ff.json").write_text("{}")
    (store.root / "events" / "franka_1arm" / "draft.txt").write_text("no")

    files = mirror.store_files(store.root)
    assert files == [
        "events/franka_1arm/abc123ff.json",
        "manifest.json",
        "tracks/franka_1arm/head.json",
    ]
    assert (store.root / ".orchestrator.lock").exists()


def test_a_root_that_is_not_a_store_is_refused(tmp_path, api):
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "orchestrator.ed25519").write_text("00" * 32 + "\n")
    with pytest.raises(ValueError, match="holds no manifest.json"):
        mirror.mirror_store(tmp_path, "org/store")
    assert api.commits == []


def test_prune_deletes_stale_paths_but_keeps_the_repository_own_files(store, api):
    api.remote = {"README.md", ".gitattributes", "manifest.json", "events/t/old.json"}
    assert mirror.mirror_store(store.root, "org/store", prune=True, message="rebuild") == 2
    ((message, ops),) = api.commits
    deleted = sorted(op.path_in_repo for op in ops if type(op).__name__ == "CommitOperationDelete")
    added = sorted(op.path_in_repo for op in ops if type(op).__name__ == "CommitOperationAdd")
    assert message == "rebuild"
    assert deleted == ["events/t/old.json"]
    assert added == ["manifest.json", "tracks/franka_1arm/head.json"]


def test_a_full_mirror_uploads_the_store_and_nothing_beside_it(store, api):
    (store.root / "keys").mkdir()
    (store.root / "keys" / "orchestrator.ed25519").write_text("00" * 32 + "\n")
    assert mirror.mirror_store(store.root, "org/store") == 2
    ((_, ops),) = api.commits
    assert sorted(op.path_in_repo for op in ops) == [
        "manifest.json",
        "tracks/franka_1arm/head.json",
    ]


def test_push_adds_exactly_what_a_publish_touched(store, api):
    assert mirror.mirror_store(store.root, "org/store", files=store.drain_touched()) == 2
    ((_, ops),) = api.commits
    assert sorted(op.path_in_repo for op in ops) == [
        "manifest.json",
        "tracks/franka_1arm/head.json",
    ]


def test_an_empty_directory_never_empties_the_repository(tmp_path, api):
    """--prune deletes every remote path the local store lacks; a wrong root must not reach it."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="holds no manifest.json"):
        mirror.mirror_store(empty, "org/store", prune=True)
    assert api.commits == []


def test_the_cli_mirrors_a_store(store, api, capsys):
    from vector_orchestrator.cli import main

    assert main(["store", "mirror", str(store.root), "--repo", "org/store", "--prune"]) == 0
    assert capsys.readouterr().out.strip() == "mirrored 2 files to org/store"
    assert len(api.commits) == 1
