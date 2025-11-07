import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # pragma: no cover - optional dependency for tests
    import colorama  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed only when colorama missing
    dummy = types.ModuleType("colorama")
    dummy.Fore = types.SimpleNamespace(RED="", YELLOW="", GREEN="", BLUE="")
    dummy.Style = types.SimpleNamespace(RESET_ALL="")

    def _noop_init(*args, **kwargs):
        return None

    dummy.init = _noop_init
    sys.modules["colorama"] = dummy

import repo_status_checker as rsc


@pytest.fixture()
def git_env(tmp_path: Path) -> dict:
    env = os.environ.copy()
    home = tmp_path / "home"
    home.mkdir()
    env.update(
        {
            "GIT_AUTHOR_NAME": "CI Tester",
            "GIT_AUTHOR_EMAIL": "ci@example.com",
            "GIT_COMMITTER_NAME": "CI Tester",
            "GIT_COMMITTER_EMAIL": "ci@example.com",
            "HOME": str(home),
        }
    )
    return env


def run_git(path: Path, args: list[str], env: dict) -> None:
    subprocess.run(["git", *args], cwd=path, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def init_bare_repo(path: Path, env: dict) -> None:
    subprocess.run(["git", "init", "--bare", str(path)], cwd=path.parent, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def create_clean_repo(path: Path, env: dict) -> Path:
    try:
        run_git(path, ["init", "-b", "main"], env)
    except subprocess.CalledProcessError:
        run_git(path, ["init"], env)
        run_git(path, ["checkout", "-b", "main"], env)
    (path / "README.md").write_text("initial\n", encoding="utf-8")
    run_git(path, ["add", "README.md"], env)
    run_git(path, ["commit", "-m", "initial"], env)

    remote_path = path.parent / f"{path.name}_remote.git"
    init_bare_repo(remote_path, env)
    run_git(path, ["remote", "add", "origin", str(remote_path)], env)
    run_git(path, ["push", "-u", "origin", "HEAD"], env)
    return remote_path


@pytest.fixture()
def clean_repo(tmp_path: Path, git_env: dict) -> tuple[Path, Path]:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    remote = create_clean_repo(repo_path, git_env)
    return repo_path, remote


def test_get_repo_status_clean_repo(clean_repo: tuple[Path, Path]) -> None:
    repo_path, _ = clean_repo
    status, extra = rsc.get_repo_status(str(repo_path))
    assert status == "clean"
    assert extra is None


def test_get_repo_status_untracked_file(clean_repo: tuple[Path, Path]) -> None:
    repo_path, _ = clean_repo
    (repo_path / "new.txt").write_text("temp\n", encoding="utf-8")
    status, _ = rsc.get_repo_status(str(repo_path))
    assert status == "untracked"


def test_get_repo_status_uncommitted_changes(clean_repo: tuple[Path, Path]) -> None:
    repo_path, _ = clean_repo
    readme = repo_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "more\n", encoding="utf-8")
    status, _ = rsc.get_repo_status(str(repo_path))
    assert status == "uncommitted"


def test_get_repo_status_unpushed_commits_reports_authors(clean_repo: tuple[Path, Path], git_env: dict) -> None:
    repo_path, _ = clean_repo
    (repo_path / "feature.txt").write_text("feature\n", encoding="utf-8")
    run_git(repo_path, ["add", "feature.txt"], git_env)
    run_git(repo_path, ["commit", "-m", "feature"], git_env)

    status, extra = rsc.get_repo_status(str(repo_path))
    assert status == "unpushed_ahead"
    assert extra == {"CI Tester <ci@example.com>"}


def test_scan_repos_non_recursive_only_top_level(tmp_path: Path, git_env: dict) -> None:
    root = tmp_path / "root"
    root.mkdir()

    top_repo = root / "top"
    top_repo.mkdir()
    create_clean_repo(top_repo, git_env)

    nested_parent = root / "nested"
    nested_parent.mkdir()
    nested_repo = nested_parent / "child"
    nested_repo.mkdir()
    create_clean_repo(nested_repo, git_env)

    statuses, _, _ = rsc.scan_repos(str(root), recursive=False)
    clean_repos = statuses.get("clean", [])
    assert str(top_repo) in clean_repos
    assert str(nested_repo) not in clean_repos
