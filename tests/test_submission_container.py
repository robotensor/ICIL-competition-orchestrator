"""The sandbox for real: Docker, the base image and the replay example. `pytest -m container`.

Builds `docker/policy-base` (Docker's cache makes a rebuild seconds) and the replay example's
image, then looks around from inside a running policy container with `docker exec`: the network,
the root filesystem, the mounts, the user, the limits. Needs Docker on the host; the GPU test
needs the nvidia runtime and one GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from icil_orchestrator.cli import main
from icil_orchestrator.ids import is_commit_sha
from icil_orchestrator.submissions.check import check_submission
from icil_orchestrator.submissions.checks import check_repository
from icil_orchestrator.submissions.container import (
    SHARED_DIR_BYTES,
    SHARED_DIR_INODES,
    PolicyContainer,
    is_shared_mount,
)
from icil_orchestrator.submissions.docker import CONTAINER_LABEL, Docker, DockerError
from icil_orchestrator.submissions.fetch import LocalFetcher, RepoCache
from icil_orchestrator.submissions.image import build_base_image, build_submission_image

pytestmark = pytest.mark.container

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "packages/icil-policy/examples/replay_policy"


@pytest.fixture(scope="module")
def docker():
    client = Docker()
    try:
        client.image_id("hello-world:nonexistent")
    except DockerError as exc:
        pytest.skip(str(exc))
    return client


@pytest.fixture(scope="module")
def base(docker, spec):
    return build_base_image(docker, spec, REPO_ROOT)


@pytest.fixture(scope="module")
def cache(spec, tmp_path_factory):
    return RepoCache(tmp_path_factory.mktemp("cache"), spec.submission["max_repo_bytes"])


@pytest.fixture(scope="module")
def replay(docker, spec, base, cache):
    """The replay example fetched, checked and built: `(fetched, manifest, image)`."""
    fetcher = LocalFetcher(cache, EXAMPLE)
    fetched = fetcher.fetch(fetcher.resolve("local/replay_policy", "main"))
    manifest = check_repository(fetched.root, spec)
    image = build_submission_image(
        docker, spec, fetched.root, manifest, fetched.resolved.ref, base.base
    )
    return fetched, manifest, image


def containers(docker) -> list[str]:
    done = docker._run(
        ["ps", "--all", "--filter", f"label={CONTAINER_LABEL}", "--format", "{{.Names}}"]
    )
    return done.stdout.split()


@pytest.fixture
def running(spec, docker, replay, tmp_path):
    """A policy container serving the replay example, past `hello`, with no GPU."""
    _, _, image = replay
    container = PolicyContainer(
        spec, docker, image.tag, name="icil-policy-test-running", socket_dir=tmp_path / "s", gpus=0
    )
    try:
        container.hello(spec.budgets["policy_start_seconds"])
        yield container
    finally:
        container.close()


def inside(container: PolicyContainer, code: str) -> subprocess.CompletedProcess:
    return container.exec(["python", "-c", code], timeout_s=60)


# -- the whole path -----------------------------------------------------------------------------


def test_the_base_image_is_the_specs_name_with_python_and_icil_policy(docker, spec, base):
    assert base.image_id.startswith("sha256:") and base.base.digest == base.image_id
    assert base.base.name == spec.submission["base_image"]["name"]
    assert docker.image_id(base.base.tag) == base.image_id
    done = docker._run(
        [
            "run",
            "--rm",
            "--user",
            spec.submission["sandbox"]["user"],
            base.base.tag,
            "python",
            "-c",
            "import sys, icil_policy; print(sys.version_info[:2], icil_policy.__version__)",
        ]
    )
    assert done.stdout.strip().startswith("(3, 10) ")


def test_submission_check_on_the_replay_example_resolves_builds_and_says_hello(
    spec, docker, base, cache, tmp_path, capsys
):
    """Acceptance criterion 1, with a local directory in the Hub's place: no token on this host
    can push the example to a test repository, and the Hub path itself is `resolve`/`fetch`."""
    code = main(
        [
            "submission",
            "check",
            "local/replay_policy@main",
            "--local",
            str(EXAMPLE),
            "--cache",
            str(cache.root),
            "--base-image",
            base.base.digest,
            "--work",
            str(tmp_path / "work"),
            "--gpus",
            "0",
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["verdict"] == "accepted", report
    assert [s["status"] for s in report["steps"]] == ["ok"] * 6
    assert is_commit_sha(report["sha"]) and report["hello"]["action_type"] == "qpos"
    # The record names the resolved commit and the base image digest.
    assert report["side"]["revision"] == report["sha"]
    assert report["side"]["base_image"] == base.base.digest
    assert report["side"]["image"].startswith("sha256:")
    assert report["listening_after_s"] < spec.budgets["policy_start_seconds"]
    assert "icil-policy-" not in " ".join(containers(docker)), "the container is gone"
    # --work keeps the log after the shared directory's tmpfs is released.
    shared = tmp_path / "work" / "policy"
    assert not is_shared_mount(shared) and "listening on" in (shared / "policy.log").read_text()
    if os.geteuid() == 0:
        assert "(a tmpfs)" in report["steps"][4]["detail"]


# -- from inside --------------------------------------------------------------------------------


def test_no_tcp_connection_and_no_dns_from_inside(running):
    tcp = inside(
        running,
        "import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=5)",
    )
    assert tcp.returncode != 0, tcp.stdout
    assert "unreachable" in tcp.stderr.lower() or "OSError" in tcp.stderr, tcp.stderr
    dns = inside(running, "import socket\nprint(socket.getaddrinfo('huggingface.co', 443))")
    assert dns.returncode != 0 and "gaierror" in dns.stderr, dns.stderr
    interfaces = inside(running, "import os\nprint(sorted(os.listdir('/sys/class/net')))")
    assert interfaces.stdout.strip() == "['lo']", interfaces.stdout


def test_writing_outside_tmp_fails_and_inside_it_works(running):
    for path in ("/submission/x", "/usr/lib/x", "/x", "/opt/x", "/etc/x", "/root/x", "/var/x"):
        done = inside(running, f"open({path!r}, 'w').write('x')")
        refused = ("Read-only file system", "Permission denied")  # /root is 0700 besides
        assert done.returncode != 0 and any(r in done.stderr for r in refused), (path, done.stderr)
    done = inside(running, "open('/tmp/x', 'w').write('x'); print(open('/tmp/x').read())")
    assert done.returncode == 0 and done.stdout.strip() == "x", done.stderr
    # /run/icil is the socket directory: writable on purpose, and this container's alone.
    done = inside(running, "import os\nprint(sorted(os.listdir('/run/icil')))")
    assert done.stdout.strip() == "['policy.log']", done.stdout


def test_the_shared_directory_is_a_tmpfs_the_policy_can_fill_and_nothing_else(running):
    """The one writable place shared with the host is bounded: past SHARED_DIR_BYTES or
    SHARED_DIR_INODES a write is ENOSPC, and the host's directory holds no more than that."""
    if not running.bounded:
        pytest.skip("not root: the shared directory is a plain directory here")
    mounts = inside(running, "print(open('/proc/mounts').read())").stdout
    (ours,) = [line for line in mounts.splitlines() if " /run/icil " in line]
    assert ours.split()[2] == "tmpfs" and f"size={SHARED_DIR_BYTES >> 10}k" in ours, ours
    filled = inside(
        running,
        "with open('/run/icil/big', 'wb') as f:\n"
        f"    for _ in range({(SHARED_DIR_BYTES >> 20) + 8}): f.write(b'x' * (1 << 20))",
    )
    assert filled.returncode != 0 and "No space left on device" in filled.stderr, filled.stderr
    many = inside(
        running,
        f"for i in range({SHARED_DIR_INODES + 8}):\n    open(f'/run/icil/f{{i}}', 'w').close()",
    )
    assert many.returncode != 0 and "No space left on device" in many.stderr, many.stderr
    entries = list(running.socket_dir.iterdir())
    assert sum(p.stat().st_size for p in entries if p.is_file()) <= SHARED_DIR_BYTES
    assert len(entries) <= SHARED_DIR_INODES
    assert is_shared_mount(running.socket_dir), "on the host, it is our tmpfs"


def test_neither_the_store_nor_another_socket_directory_is_visible(
    spec, docker, replay, tmp_path, monkeypatch
):
    store = tmp_path / "store"
    (store / "tracks").mkdir(parents=True)
    (store / "manifest.json").write_text("{}")
    other = tmp_path / "other-side"
    other.mkdir()
    (other / "policy.sock").write_text("the king's socket")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_that_must_stay_on_the_host")
    monkeypatch.setenv("ICIL_LIVE_TOKEN", "live_secret_that_must_stay_on_the_host")
    _, _, image = replay
    with PolicyContainer(
        spec, docker, image.tag, name="icil-policy-test-blind", socket_dir=tmp_path / "s", gpus=0
    ) as container:
        container.hello(spec.budgets["policy_start_seconds"])
        for path in (store, other, tmp_path):
            done = inside(container, f"import os\nprint(os.listdir({str(path)!r}))")
            assert done.returncode != 0 and "FileNotFoundError" in done.stderr, (path, done.stdout)
        mounts = inside(container, "print(open('/proc/mounts').read())").stdout
        ours = [line for line in mounts.splitlines() if "/run/icil" in line]
        assert len(ours) == 1 and str(tmp_path) not in mounts.replace(str(tmp_path / "s"), "")
        assert "store" not in mounts and "other-side" not in mounts
        env = inside(container, "import os\nprint(sorted(os.environ))").stdout
        assert "HF_TOKEN" not in env and "ICIL_LIVE_TOKEN" not in env, env
        # (That the server drops the authkey from its own environment before the policy is
        # built is the protocol's own test: /proc/<pid>/environ shows the exec-time block.)


def test_the_user_and_the_limits_are_the_specs(spec, docker, running):
    sandbox = spec.submission["sandbox"]
    uid, _, gid = sandbox["user"].partition(":")
    ids = inside(running, "import os\nprint(os.getuid(), os.getgid())").stdout.split()
    assert ids == [uid, gid or uid]
    caps = inside(running, "print(open('/proc/self/status').read())").stdout
    assert "CapEff:\t0000000000000000" in caps and "NoNewPrivs:\t1" in caps
    # The limits are read from the container's cgroup v2 files; on a v1 host they are elsewhere,
    # and the test says so rather than passing with nothing checked.
    version = docker._run(["info", "--format", "{{.CgroupVersion}}"]).stdout.strip()
    if version != "2":
        pytest.skip(f"the daemon runs cgroup v{version}; the limits are read from v2 files")
    limits = inside(
        running,
        "for f in ('pids.max', 'memory.max', 'memory.swap.max', 'cpu.max'):\n"
        "    print(f, open('/sys/fs/cgroup/' + f).read().strip())",
    )
    assert limits.returncode == 0, limits.stderr
    lines = dict(line.split(" ", 1) for line in limits.stdout.strip().splitlines())
    assert lines["pids.max"] == str(sandbox["pids"])
    assert lines["memory.max"] == str(sandbox["memory_bytes"])
    assert lines["memory.swap.max"] == "0", "the spec's bytes are the total: no swap"
    quota, period = lines["cpu.max"].split()
    assert int(quota) / int(period) == sandbox["cpus"]


def test_the_gpu_is_there_when_the_spec_asks_for_one(spec, docker, replay, tmp_path):
    _, _, image = replay
    with PolicyContainer(
        spec, docker, image.tag, name="icil-policy-test-gpu", socket_dir=tmp_path / "s"
    ) as container:
        container.hello(spec.budgets["policy_start_seconds"])
        done = inside(
            container,
            "import os\nprint(sorted(d for d in os.listdir('/dev') if d.startswith('nvidia')))",
        )
        assert done.returncode == 0 and "nvidia0" in done.stdout and "nvidiactl" in done.stdout
        assert container.exec(["nvidia-smi", "-L"]).returncode == 0
        assert inside(container, "open('/x', 'w')").returncode != 0, "still read-only"


# -- rejections ---------------------------------------------------------------------------------


def test_a_missing_class_is_rejected_at_hello_and_no_container_is_left(
    spec, docker, base, cache, tmp_path
):
    broken = shutil.copytree(EXAMPLE, tmp_path / "broken")
    (broken / "icil.yaml").write_text("api: 1\npolicy: replay.policy:NoSuchPolicy\n")
    report = check_submission(
        spec,
        "local/broken",
        "main",
        fetcher=LocalFetcher(cache, broken),
        docker=docker,
        work_dir=tmp_path / "work",
        base_digest=base.base.digest,
        gpus=0,
    )
    assert report.verdict == "rejected" and report.failed_step.name == "hello", report.as_dict()
    assert "has no attribute 'NoSuchPolicy'" in report.failed_step.detail
    assert report.side()["revision"] == report.sha and report.side()["base_image"] == (
        base.base.digest
    )
    assert not any(n.startswith(f"icil-policy-{report.key}") for n in containers(docker))


def test_requirements_that_do_not_install_are_rejected_at_build_and_nothing_runs(
    spec, docker, base, cache, tmp_path
):
    broken = shutil.copytree(EXAMPLE, tmp_path / "broken")
    (broken / "requirements.txt").write_text("icil-no-such-package==99.0\n")
    report = check_submission(
        spec,
        "local/broken",
        "main",
        fetcher=LocalFetcher(cache, broken),
        docker=docker,
        work_dir=tmp_path / "work",
        base_digest=base.base.digest,
        gpus=0,
    )
    assert report.verdict == "rejected" and report.failed_step.name == "build", report.as_dict()
    assert "installing requirements.txt failed" in report.failed_step.detail
    assert "icil-no-such-package" in report.failed_step.detail
    assert [s.status for s in report.steps] == ["ok", "ok", "ok", "rejected", "skipped", "skipped"]
    assert not any(n.startswith(f"icil-policy-{report.key}") for n in containers(docker))
    assert docker.image_id(f"icil-submission:{report.key}-{report.sha}") is None


def test_requirements_that_never_finish_installing_are_rejected_at_build_in_time(
    spec, docker, base, cache, tmp_path
):
    """A setup.py that never returns: the build is cut at its timeout, the submission is
    rejected at build, and no image of it is left."""
    stalled = shutil.copytree(EXAMPLE, tmp_path / "stalled")
    (stalled / "requirements.txt").write_text("./stall\n")
    (stalled / "stall").mkdir()
    (stalled / "stall" / "setup.py").write_text(
        "import time\ntime.sleep(10**6)\nfrom setuptools import setup\nsetup(name='stall')\n"
    )
    started = time.monotonic()
    report = check_submission(
        spec,
        "local/stalled",
        "main",
        fetcher=LocalFetcher(cache, stalled),
        docker=docker,
        work_dir=tmp_path / "work",
        base_digest=base.base.digest,
        gpus=0,
        build_timeout_s=20.0,
    )
    took = time.monotonic() - started
    assert report.verdict == "rejected" and report.failed_step.name == "build", report.as_dict()
    assert report.failed_step.detail == "installing requirements.txt did not finish within 20s"
    assert 20 <= took < 60, took
    assert docker.image_id(f"icil-submission:{report.key}-{report.sha}") is None
    assert not any(n.startswith(f"icil-policy-{report.key}") for n in containers(docker))
