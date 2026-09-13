import os
import textwrap

import pytest

from icil_policy import ManifestError, manifest


def write(root, text, name="icil.yaml"):
    path = root / name
    path.write_text(textwrap.dedent(text))
    return path


def test_the_smallest_manifest_is_api_and_policy(tmp_path):
    loaded = manifest.load(write(tmp_path, "api: 1\npolicy: my_policy:Policy\n"))
    assert loaded.api == 1
    assert loaded.policy == "my_policy:Policy"
    assert (loaded.module, loaded.attribute) == ("my_policy", "Policy")
    assert loaded.kwargs == {}
    assert loaded.requirements is None and loaded.requirements_path is None
    assert loaded.benchmarks == ()
    assert loaded.root == tmp_path


def test_every_key_is_read(tmp_path):
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "requirements.txt").write_text("numpy\n")
    loaded = manifest.load(
        write(
            tmp_path,
            """
            api: 1
            policy: pkg.sub.module:MyPolicy
            kwargs: {checkpoint: weights/model.pt, horizon: 8, gains: [1.0, 2.0]}
            requirements: env/requirements.txt
            benchmarks: [robotwin, libero]
            """,
        )
    )
    assert loaded.policy == "pkg.sub.module:MyPolicy"
    assert loaded.kwargs == {"checkpoint": "weights/model.pt", "horizon": 8, "gains": [1.0, 2.0]}
    assert loaded.requirements_path == tmp_path / "env" / "requirements.txt"
    assert loaded.benchmarks == ("robotwin", "libero")


def test_a_relative_path_is_made_absolute(tmp_path, monkeypatch):
    write(tmp_path, "api: 1\npolicy: a:B\n")
    monkeypatch.chdir(tmp_path)
    assert manifest.load("icil.yaml").path == tmp_path / "icil.yaml"


def test_every_problem_is_listed_in_one_error(tmp_path):
    path = write(
        tmp_path,
        """
        api: 2
        policy: not a policy
        kwargs: [1, 2]
        requirements: ../outside.txt
        benchmarks: robotwin
        entrypoint: main.py
        """,
    )
    with pytest.raises(ManifestError) as caught:
        manifest.load(path)
    problems = caught.value.problems
    assert len(problems) == 6, problems
    text = str(caught.value)
    for expected in (
        "unknown key(s) 'entrypoint'",
        "api: must be 1, not 2",
        "policy: must be module:Class",
        "kwargs: must be a mapping",
        "leaves the repository",
        "benchmarks: must be a list",
    ):
        assert expected in text
    assert str(path) in text


def test_api_and_policy_are_required(tmp_path):
    with pytest.raises(ManifestError) as caught:
        manifest.load(write(tmp_path, "kwargs: {}\n"))
    assert caught.value.problems == ["api: required", "policy: required, as module:Class"]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("api: '1'\npolicy: a:B\n", "api: must be 1"),
        ("api: true\npolicy: a:B\n", "api: must be 1"),
        ("api: 1.0\npolicy: a:B\n", "api: must be 1"),
        ("api: 1\npolicy: a\n", "policy: must be module:Class"),
        ("api: 1\npolicy: a:B.c\n", "policy: must be module:Class"),
        ("api: 1\npolicy: a/b:C\n", "policy: must be module:Class"),
        ("api: 1\npolicy: 1a:C\n", "policy: must be module:Class"),
        ("api: 1\npolicy: [a, B]\n", "policy: must be module:Class"),
        ("api: 1\npolicy: |\n  a:B\n", "policy: must be module:Class"),
        ("api: 1\npolicy: a:B\nkwargs: {1: x}\n", "kwargs: keys must be identifiers"),
        ("api: 1\npolicy: a:B\nkwargs: {not-an-id: x}\n", "kwargs: keys must be identifiers"),
        ("api: 1\npolicy: a:B\nrequirements: 3\n", "requirements: must be a path"),
        ("api: 1\npolicy: a:B\nrequirements: /etc/passwd\n", "must be relative"),
        ("api: 1\npolicy: a:B\nrequirements: ~/r.txt\n", "must be relative"),
        ("api: 1\npolicy: a:B\nrequirements: sub/../../r.txt\n", "leaves the repository"),
        ("api: 1\npolicy: a:B\nrequirements: missing.txt\n", "is not a file"),
        ("api: 1\npolicy: a:B\nrequirements: .\n", "is not a file"),
        ("api: 1\npolicy: a:B\nbenchmarks: [robotwin, 3]\n", "benchmarks: must be a list"),
        ("api: 1\npolicy: a:B\nbenchmarks: ['']\n", "benchmarks: must be a list"),
    ],
)
def test_a_bad_value_is_named(tmp_path, text, problem):
    with pytest.raises(ManifestError) as caught:
        manifest.load(write(tmp_path, text))
    assert len(caught.value.problems) == 1
    assert problem in caught.value.problems[0]


def test_requirements_may_not_escape_through_a_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "requirements.txt").write_text("evil\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    os.symlink(outside, repo / "link")
    with pytest.raises(ManifestError, match="leaves the repository"):
        manifest.load(write(repo, "api: 1\npolicy: a:B\nrequirements: link/requirements.txt\n"))


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", "must be a mapping, not NoneType"),
        ("- api: 1\n", "must be a mapping, not list"),
        ("api: 1\npolicy: [\n", "is not valid YAML"),
        ("api: 1\napi: 1\npolicy: a:B\n", "'api' is given twice"),
        ("api: 1\npolicy: a:B\nkwargs: {x: 1, x: 2}\n", "'x' is given twice"),
    ],
)
def test_a_file_that_is_not_a_manifest_is_refused(tmp_path, text, problem):
    with pytest.raises(ManifestError, match=problem):
        manifest.load(write(tmp_path, text))


def test_merge_keys_share_kwargs_and_a_key_given_beside_a_merge_overrides_it(tmp_path):
    text = """
    api: 1
    policy: a:B
    kwargs:
      base: &base {lr: 1, horizon: 8}
      extra: &extra {seed: 3}
      one: {<<: *base, horizon: 16}
      both:
        <<: [*base, *extra]
        lr: 2
    """
    loaded = manifest.load(write(tmp_path, text))
    assert loaded.kwargs["one"] == {"lr": 1, "horizon": 16}
    assert loaded.kwargs["both"] == {"lr": 2, "horizon": 8, "seed": 3}
    with pytest.raises(ManifestError, match="'lr' is given twice"):
        manifest.load(write(tmp_path, text.replace("lr: 2", "lr: 2\n        lr: 3")))


def test_a_python_tag_is_refused_and_never_run(tmp_path):
    marker = tmp_path / "ran"
    text = f"api: 1\npolicy: a:B\nkwargs: {{x: !!python/object/apply:os.mkdir ['{marker}']}}\n"
    with pytest.raises(ManifestError, match="is not valid YAML"):
        manifest.load(write(tmp_path, text))
    assert not marker.exists()


def test_a_missing_file_is_a_manifest_error(tmp_path):
    with pytest.raises(ManifestError, match="cannot be read"):
        manifest.load(tmp_path / "icil.yaml")
