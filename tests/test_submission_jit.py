"""Compiling at run time inside the policy sandbox, for real. `pytest -m container`.

The base image carries gcc, g++, make, Python's headers and the CUDA toolkit's nvcc, and the
sandbox's /tmp is a tmpfs that may run what is written there - nosuid, nodev and no larger than
`sandbox.tmpfs_bytes` - with HOME, TMPDIR and the compilers' caches pointed into it. These tests
serve competitor repositories that compile (`tests/fixtures/jit_policy`, and under `slow`
`tests/fixtures/torch_compile_policy`) through the real sandbox and `RemotePolicy`, and check that
the spec's tmpfs is the only place code a policy writes can run from: elsewhere the root is
read-only, or the mount (/dev/shm, the shared socket directory) is noexec. The network, the user,
the limits and what a policy cannot see are `test_submission_container.py`'s.

Containers are named `icil-jit-*`, and the images built here are removed when the module ends.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from icil_orchestrator.submissions.checks import check_repository
from icil_orchestrator.submissions.container import (
    SOCKET_DIR,
    PolicyContainer,
    policy_environment,
)
from icil_orchestrator.submissions.docker import Docker, DockerError
from icil_orchestrator.submissions.fetch import LocalFetcher, RepoCache
from icil_orchestrator.submissions.image import (
    build_base_image,
    build_submission_image,
    sandbox_user,
)
from icil_policy.errors import PolicyUnavailable

pytestmark = pytest.mark.container

REPO_ROOT = Path(__file__).resolve().parents[1]
JIT_POLICY = REPO_ROOT / "tests/fixtures/jit_policy"
TORCH_POLICY = REPO_ROOT / "tests/fixtures/torch_compile_policy"

#: Run inside a container: compile a one-line C function into argv[1] and call it from there.
COMPILE_AND_LOAD = """
import ctypes, os, subprocess, sys
target = sys.argv[1]
source = os.path.join(os.environ["TMPDIR"], "one.c")
with open(source, "w") as f:
    f.write("int cjit_one(void) { return 1; }\\n")
done = subprocess.run(
    ["gcc", "-shared", "-fPIC", "-o", target, source], capture_output=True, text=True
)
if done.returncode:
    sys.exit("compile failed: " + done.stderr)
print(ctypes.CDLL(target).cjit_one())
"""

#: Run inside a container: the mount points where the user it runs as can put code and run it,
#: as a sorted JSON list. A mount counts when /proc/mounts has it neither `ro` nor `noexec` and, on
#: that filesystem alone (as `find -xdev`), a directory takes a new file or a file of the user's own
#: can be written.
WRITABLE_AND_EXECUTABLE = r"""
import json, os, stat
mine = os.getuid()
found = set()
for line in open("/proc/mounts"):
    fields = line.split()
    point = fields[1].encode().decode("unicode_escape")
    flags = fields[3].split(",")
    if "ro" in flags or "noexec" in flags:
        continue
    try:
        top = os.lstat(point)
    except OSError:
        continue
    if not stat.S_ISDIR(top.st_mode):
        if top.st_uid == mine and os.access(point, os.W_OK):
            found.add(point)
        continue
    for root, dirs, files in os.walk(point):
        if os.access(root, os.W_OK | os.X_OK):
            found.add(point)
            break
        kept = []
        for name in dirs:
            try:
                if os.lstat(os.path.join(root, name)).st_dev == top.st_dev:
                    kept.append(name)
            except OSError:
                pass
        dirs[:] = kept
        for name in files:
            path = os.path.join(root, name)
            try:
                info = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode) and info.st_uid == mine and os.access(path, os.W_OK):
                found.add(point)
print(json.dumps(sorted(found)))
"""


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
def build(docker, spec, base, cache):
    """Build a competitor directory's image as `submission check` does; each is removed after."""
    built = []

    def build_image(directory: Path, repo: str):
        fetcher = LocalFetcher(cache, directory)
        fetched = fetcher.fetch(fetcher.resolve(repo, "main"))
        manifest = check_repository(fetched.root, spec)
        image = build_submission_image(
            docker, spec, fetched.root, manifest, fetched.resolved.ref, base.base
        )
        built.append(image.tag)
        return image

    yield build_image
    for tag in built:
        docker.remove_image(tag)


@pytest.fixture(scope="module")
def jit_image(build):
    return build(JIT_POLICY, "local/jit_policy")


@pytest.fixture
def jit(spec, docker, jit_image, tmp_path):
    """The C-compiling policy in its container, past `hello`, with no GPU."""
    container = PolicyContainer(
        spec, docker, jit_image.tag, name="icil-jit-cjit", socket_dir=tmp_path / "s", gpus=0
    )
    try:
        container.hello(spec.budgets["policy_start_seconds"])
        yield container
    finally:
        container.close()


def inside(container: PolicyContainer, *argv: str, timeout_s: float = 120):
    return container.exec(list(argv), timeout_s=timeout_s)


def python(container: PolicyContainer, code: str, *args: str):
    return inside(container, "python", "-c", code, *args)


# -- compiling ----------------------------------------------------------------------------------


def test_a_policy_that_compiles_c_on_its_first_act_answers_with_it(spec, jit):
    """hello, prompt, reset and act through `RemotePolicy`: the first act writes C to $TMPDIR,
    compiles it with gcc -shared into the cache on /tmp, loads it with ctypes and answers with
    what it computes."""
    env = policy_environment(spec)
    policy = jit.session
    assert policy is not None and policy.action_type == "qpos"
    policy.set_demonstration({"actions": np.zeros((4, 16))}, {"fps": 10})
    policy.reset(1234)
    empty = inside(jit, "ls", "-A", env["XDG_CACHE_HOME"])
    assert empty.returncode == 0 and empty.stdout == "", "made at start, empty until act"
    for obs in (np.linspace(-2.0, 2.0, 16), np.arange(16.0)):
        action = policy.act({"qpos": obs})["action"]
        np.testing.assert_allclose(action, 3.0 * obs + np.arange(16) + 0.5, rtol=0, atol=1e-12)
    library = f"{env['XDG_CACHE_HOME']}/cjit/cjit.so"
    assert inside(jit, "test", "-f", f"{env['TMPDIR']}/cjit.c").returncode == 0
    assert inside(jit, "test", "-f", library).returncode == 0
    maps = inside(jit, "cat", "/proc/1/maps")
    assert library in maps.stdout, "the server, the container's first process, has it mapped"


def test_the_same_compile_anywhere_but_tmp_is_not_written_or_not_run(
    spec, docker, build, jit, tmp_path
):
    """The policy compiling into /submission - its own checkout, owned by its user - fails at act,
    and the served session goes on. By hand: the root and the image's directories refuse the
    shared object, /tmp takes it and loads it, and /dev/shm (the other tmpfs Docker gives a
    container) and /run/icil (the socket directory, which the policy may write) take it and refuse
    to run it."""
    outside = shutil.copytree(JIT_POLICY, tmp_path / "outside")
    (outside / "icil.yaml").write_text(
        "api: 1\npolicy: cjit.policy:CJitPolicy\nkwargs:\n  build_dir: /submission\n"
    )
    image = build(outside, "local/jit_outside")
    with PolicyContainer(
        spec, docker, image.tag, name="icil-jit-outside", socket_dir=tmp_path / "o", gpus=0
    ) as container:
        container.hello(spec.budgets["policy_start_seconds"])
        with pytest.raises(PolicyUnavailable) as info:
            container.session.act({"qpos": np.zeros(16)})
        message = str(info.value)
        assert "gcc failed" in message and "Read-only file system" in message, message
        assert "/submission/cjit.so" in message
        container.session.reset(1)  # an error reply leaves the session usable

    for target in ("/one.so", "/submission/one.so", "/usr/local/lib/one.so", "/opt/one.so"):
        done = python(jit, COMPILE_AND_LOAD, target)
        assert done.returncode != 0 and "compile failed" in done.stderr, (target, done.stderr)
        assert "Read-only file system" in done.stderr, (target, done.stderr)
    scratch = python(jit, COMPILE_AND_LOAD, f"{policy_environment(spec)['TMPDIR']}/one.so")
    assert scratch.returncode == 0 and scratch.stdout.strip() == "1", scratch.stderr
    assert jit.bounded, "the container tests run as root, so the socket directory is a tmpfs"
    for target in ("/dev/shm/one.so", f"{SOCKET_DIR}/one.so"):
        done = python(jit, COMPILE_AND_LOAD, target)
        assert done.returncode != 0 and "compile failed" not in done.stderr, (target, done.stderr)
        assert "failed to map segment from shared object" in done.stderr, (target, done.stderr)


def test_the_specs_tmpfs_is_the_only_place_the_policy_can_write_code_and_run_it(spec, jit):
    """Every mount inside, walked as the sandbox user: those neither read-only nor noexec where it
    can create or rewrite a file are exactly `sandbox.tmpfs` (none when `tmpfs_exec` is false)."""
    sandbox = spec.submission["sandbox"]
    found = python(jit, WRITABLE_AND_EXECUTABLE)
    assert found.returncode == 0, found.stderr
    expected = sorted(sandbox["tmpfs"]) if sandbox["tmpfs_exec"] else []
    assert json.loads(found.stdout) == expected


def test_tmp_is_a_nosuid_nodev_tmpfs_of_the_specs_size_that_runs_code(spec, jit):
    """What /proc/mounts says inside: every `sandbox.tmpfs` path a tmpfs, nosuid and nodev, exec
    as the spec says, of `tmpfs_bytes`, and the socket directory nosuid, nodev and noexec; and the
    environment and home the container starts with."""
    sandbox = spec.submission["sandbox"]
    mounts = inside(jit, "cat", "/proc/mounts").stdout.splitlines()

    def flags_at(path: str) -> list[str]:
        (line,) = [m for m in mounts if m.split()[1:2] == [path]]
        _, _, kind, options, *_ = line.split()
        assert kind == "tmpfs", line
        return options.split(",")

    for path in sandbox["tmpfs"]:
        flags = flags_at(path)
        assert "rw" in flags and "nosuid" in flags and "nodev" in flags, flags
        assert ("noexec" not in flags) is sandbox["tmpfs_exec"], flags
        assert f"size={sandbox['tmpfs_bytes'] >> 10}k" in flags, flags
    assert jit.bounded
    shared = flags_at(SOCKET_DIR)
    assert {"rw", "nosuid", "nodev", "noexec"} <= set(shared), shared
    environ = json.loads(python(jit, "import json, os\nprint(json.dumps(dict(os.environ)))").stdout)
    served = inside(jit, "cat", "/proc/1/environ").stdout.split("\0")
    for key, value in policy_environment(spec).items():
        assert environ[key] == value and f"{key}={value}" in served, key
    uid, gid = sandbox_user(spec)
    home = python(
        jit,
        "import json, os, stat\n"
        "for d in (os.environ['HOME'], os.environ['XDG_CACHE_HOME']):\n"
        "    s = os.stat(d)\n"
        "    print(json.dumps([s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode)]))",
    )
    assert [json.loads(line) for line in home.stdout.splitlines()] == [[uid, gid, 0o700]] * 2


def test_nvcc_gcc_and_the_python_headers_are_there_for_the_sandbox_user(spec, jit):
    """No GPU in this container: nvcc still reports its version and compiles a kernel."""
    uid, _ = sandbox_user(spec)
    assert python(jit, "import os; print(os.getuid())").stdout.strip() == str(uid)
    nvcc = inside(jit, "nvcc", "--version")
    assert nvcc.returncode == 0 and "Cuda compilation tools, release 12.8" in nvcc.stdout, nvcc
    for tool in ("gcc", "g++", "make"):
        assert inside(jit, tool, "--version").returncode == 0, tool
    header = python(
        jit,
        "import os, sysconfig\n"
        "print(os.path.isfile(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))",
    )
    assert header.stdout.strip() == "True", header.stderr
    kernel = inside(
        jit,
        "sh",
        "-c",
        'printf "__global__ void fill(float *x, int i) { x[i] = 1.0f; }\\n" > "$TMPDIR/k.cu"'
        ' && nvcc -c "$TMPDIR/k.cu" -o "$TMPDIR/k.o" && test -s "$TMPDIR/k.o"',
    )
    assert kernel.returncode == 0, kernel.stderr


# -- torch.compile ------------------------------------------------------------------------------


@pytest.mark.slow
def test_a_torch_compiled_act_runs_on_the_cpu_in_the_sandbox(spec, docker, build, tmp_path, capsys):
    """CPU torch installed at build time; inductor compiles C++ with g++ into
    $TORCHINDUCTOR_CACHE_DIR during hello and act runs it. The times and the image size are
    printed for the base image's README."""
    started = time.monotonic()
    image = build(TORCH_POLICY, "local/torch_compile_policy")
    build_seconds = time.monotonic() - started
    size = int(docker._run(["image", "inspect", "--format", "{{.Size}}", image.tag]).stdout)
    env = policy_environment(spec)
    with PolicyContainer(
        spec, docker, image.tag, name="icil-jit-torch", socket_dir=tmp_path / "s", gpus=0
    ) as container:
        started = time.monotonic()
        container.hello(spec.budgets["policy_start_seconds"])
        hello_seconds = time.monotonic() - started
        policy = container.session
        policy.set_demonstration({"actions": np.zeros((4, 16))}, {"fps": 10})
        policy.reset(0)
        obs = np.linspace(-3.0, 3.0, 16)
        started = time.monotonic()
        reply = policy.act({"qpos": obs})
        act_seconds = time.monotonic() - started
        expected = np.sin(obs) * 2.0 + np.cos(obs) ** 2
        np.testing.assert_allclose(reply["action"], expected, rtol=1e-12, atol=1e-12)
        found = inside(container, "find", env["TORCHINDUCTOR_CACHE_DIR"], "-name", "*.so")
        libraries = found.stdout.split()
        assert found.returncode == 0 and libraries, (found.stdout, found.stderr)
        maps = inside(container, "cat", "/proc/1/maps").stdout
        assert any(library in maps for library in libraries), "the server runs what inductor built"
    with capsys.disabled():
        print(
            f"\n[torch.compile in the sandbox] image build {build_seconds:.1f} s, "
            f"image {size} bytes, hello (import torch and compile) {hello_seconds:.1f} s, "
            f"compile {float(reply['compile_seconds']):.2f} s, "
            f"act {act_seconds * 1000:.1f} ms (compiled call {float(reply['act_seconds']) * 1000:.2f} ms)"
        )
