#!/usr/bin/env python3
"""
Git Repository Status Checker
Scans subdirectories for Git repositories and reports their status, including:
- bare repos
- detached HEAD
- uncommitted / untracked changes
- no upstream
- diverged / ahead (unpushed) / behind / up-to-date
- authors of unpushed commits (for ahead repos)

Color output via colorama. Logs details to a file.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from colorama import Fore, Style, init as colorama_init


# --- Color setup ---
colorama_init(autoreset=True)
RED   = Fore.RED
YEL   = Fore.YELLOW
GRN   = Fore.GREEN
BLU   = Fore.BLUE
RST   = Style.RESET_ALL


# --- Categories: key -> (color, label) ---
CATEGORIES: Dict[str, Tuple[str, str]] = {
    "bare":            (BLU,  "Bare Git Repositories"),
    "detached_head":   (YEL,  "Repos in Detached HEAD State"),
    "uncommitted":     (RED,  "Repos with Uncommitted Changes"),
    "untracked":       (YEL,  "Repos with Untracked Files"),
    "no_upstream":     (BLU,  "Repos with No Upstream Set"),
    "diverged":        (RED,  "Repos Diverged from Upstream"),
    "unpushed_ahead":  (YEL,  "Repos with Unpushed Commits (Ahead)"),
    "behind":          (YEL,  "Repos Behind Upstream"),
    "clean":           (GRN,  "Repos Up-to-Date and Tracking Remote"),
}


def run_git(repo_path: str, args: List[str]) -> Tuple[str, str, int]:
    """Run a git command in repo_path and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        logging.exception("Error running git %s in %s", " ".join(args), repo_path)
        return "", str(e), 1


def is_git_dir(path: str) -> bool:
    """Detect a working-tree repo by existence of .git (dir or file for worktrees)."""
    git_path = os.path.join(path, ".git")
    return os.path.isdir(git_path) or os.path.isfile(git_path)


def is_bare_repo(path: str) -> bool:
    """Return True if repo is bare."""
    out, _, _ = run_git(path, ["rev-parse", "--is-bare-repository"])
    return out.lower() == "true"


def current_branch_or_head(path: str) -> str:
    """Return current branch name or 'HEAD' if detached."""
    out, _, _ = run_git(path, ["rev-parse", "--abbrev-ref", "HEAD"])
    return out  # 'HEAD' when detached


def has_uncommitted(path: str) -> bool:
    """True if there are staged/unstaged changes (excluding untracked)."""
    out, _, _ = run_git(path, ["status", "--porcelain"])
    return any(line and not line.startswith("??") for line in out.splitlines())


def has_untracked(path: str) -> bool:
    """True if there are untracked files."""
    out, _, _ = run_git(path, ["status", "--porcelain"])
    return any(line.startswith("??") for line in out.splitlines())


def upstream_ref(path: str, branch: str) -> Optional[str]:
    """Return @{upstream} name or None if not set."""
    out, err, rc = run_git(path, ["rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"])
    if rc != 0 or "fatal" in err.lower():
        return None
    return out


def ahead_behind(path: str, upstream: str) -> Tuple[int, int]:
    """
    Return (ahead, behind) relative to upstream.
    ahead: commits in HEAD not in upstream
    behind: commits in upstream not in HEAD
    """
    ahead_out, _, _  = run_git(path, ["rev-list", "--count", f"{upstream}..HEAD"])
    behind_out, _, _ = run_git(path, ["rev-list", "--count", f"HEAD..{upstream}"])
    try:
        return int(ahead_out or 0), int(behind_out or 0)
    except ValueError:
        return 0, 0


def unpushed_authors(path: str, upstream: str) -> Set[str]:
    """Return authors of commits that are ahead of upstream."""
    out, _, _ = run_git(path, ["log", "--format=%an <%ae>", f"{upstream}..HEAD"])
    return {line.strip() for line in out.splitlines() if line.strip()}


def get_repo_status(path: str) -> Tuple[str, Optional[object]]:
    """
    Determine repo status.
    Returns (status_key, extra):
      - 'diverged' -> extra = (ahead:int, behind:int)
      - 'unpushed_ahead' -> extra = Set[str] authors
      - other -> extra = None
    """
    if is_bare_repo(path):
        return "bare", None

    branch = current_branch_or_head(path)
    if branch == "HEAD":
        return "detached_head", None

    if has_uncommitted(path):
        return "uncommitted", None

    if has_untracked(path):
        return "untracked", None

    up = upstream_ref(path, branch)
    if not up:
        return "no_upstream", None

    ahead, behind = ahead_behind(path, up)
    if ahead > 0 and behind > 0:
        return "diverged", (ahead, behind)
    elif ahead > 0:
        authors = unpushed_authors(path, up)
        return "unpushed_ahead", authors
    elif behind > 0:
        return "behind", None
    else:
        return "clean", None


def scan_repos(root: str, recursive: bool) -> Tuple[
    Dict[str, List[str]],
    Dict[str, Tuple[int, int]],
    Dict[str, Set[str]],
]:
    """
    Walk `root` and gather repo statuses.
    Returns:
      - repo_statuses: status_key -> [repo_paths]
      - diverged_info: repo_path -> (ahead, behind)
      - unpushed_info: repo_path -> {authors}
    """
    repo_statuses: Dict[str, List[str]] = defaultdict(list)
    diverged_info: Dict[str, Tuple[int, int]] = {}
    unpushed_info: Dict[str, Set[str]] = {}

    for dirpath, dirnames, _ in os.walk(root):
        if is_git_dir(dirpath):
            status, extra = get_repo_status(dirpath)
            repo_statuses[status].append(dirpath)

            if status == "diverged" and isinstance(extra, tuple):
                diverged_info[dirpath] = extra
            elif status == "unpushed_ahead" and isinstance(extra, set):
                unpushed_info[dirpath] = extra

            # do not descend into .git dirs
            if ".git" in dirnames:
                dirnames.remove(".git")

            # optionally stop at first level
            if not recursive:
                dirnames[:] = []
        else:
            # still avoid recursing into .git if encountered somewhere odd
            if ".git" in dirnames:
                dirnames.remove(".git")
            if not recursive:
                # only check immediate children of root
                if os.path.abspath(dirpath) != os.path.abspath(root):
                    dirnames[:] = []

    return repo_statuses, diverged_info, unpushed_info


def print_results(
    repo_statuses: Dict[str, List[str]],
    diverged_info: Dict[str, Tuple[int, int]],
    unpushed_info: Dict[str, Set[str]],
    skip_clean: bool,
) -> None:
    problems: Dict[str, List[str]] = defaultdict(list)
    clean = repo_statuses.get("clean", [])

    # Per-category listing
    for key, (color, label) in CATEGORIES.items():
        repos = repo_statuses.get(key, [])
        if not repos:
            continue
        if key == "clean" and skip_clean:
            continue

        print(f"{color}{label}{RST}")
        for repo in sorted(repos):
            if key == "diverged":
                ahead, behind = diverged_info.get(repo, (0, 0))
                print(f"  {color}{repo}{RST}  {YEL}(ahead: {ahead}, behind: {behind}){RST}")
            elif key == "unpushed_ahead":
                authors = sorted(unpushed_info.get(repo, set()))
                who = ", ".join(authors) if authors else "Unknown author(s)"
                print(f"  {color}{repo}{RST}  {YEL}(Authors: {who}){RST}")
            else:
                print(f"  {color}{repo}{RST}")

        if key != "clean":
            problems[key].extend(repos)

    # Summary
    div_count = len(repo_statuses.get("diverged", []))
    up_count  = len(repo_statuses.get("unpushed_ahead", []))
    other_problem_count = sum(
        len(v) for k, v in problems.items() if k not in ("diverged", "unpushed_ahead")
    )

    print()
    print(f"{BLU}Summary{RST}")
    if div_count:
        print(f"  {RED}Diverged: {div_count}{RST}")
    if up_count:
        print(f"  {YEL}Unpushed (ahead): {up_count}{RST}")
    if other_problem_count:
        print(f"  {RED}Other problems: {other_problem_count}{RST}")
    if not (div_count or up_count or other_problem_count):
        print(f"  {GRN}No problem repos detected.{RST}")

    if not skip_clean and clean:
        print(f"  {GRN}Clean: {len(clean)}{RST}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan subdirectories for Git repositories and report status."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="Directory to scan (default: current directory)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recurse into subdirectories (default: only immediate children)",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Do not print clean repositories",
    )
    parser.add_argument(
        "--log",
        default="git_status.log",
        help="Log file path (default: git_status.log)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo_statuses, div_info, up_info = scan_repos(args.root_dir, args.recursive)
    print_results(repo_statuses, div_info, up_info, args.skip_clean)


if __name__ == "__main__":
    main()
