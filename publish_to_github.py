import argparse
import base64
import fnmatch
import os
from pathlib import Path

import requests


PROJECT_DIR = Path(__file__).resolve().parent
GITHUB_API = "https://api.github.com"
IGNORE_PATTERNS = [
    ".env",
    "bot_memory.sqlite3",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.sqlite3",
    ".venv/*",
    "venv/*",
    "*.log",
]


def should_ignore(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    if normalized == "__pycache__":
        return True
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in IGNORE_PATTERNS)


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if should_ignore(relative):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def github_request(method: str, url: str, token: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    headers["Accept"] = "application/vnd.github+json"
    response = requests.request(method, url, headers=headers, timeout=60, **kwargs)
    return response


def require_ok(response: requests.Response, expected: tuple[int, ...]) -> dict:
    if response.status_code not in expected:
        raise SystemExit(
            f"GitHub API error {response.status_code} for {response.request.method} "
            f"{response.request.url}: {response.text}"
        )
    if response.text.strip():
        return response.json()
    return {}


def get_authenticated_user(token: str) -> dict:
    response = github_request("GET", f"{GITHUB_API}/user", token)
    return require_ok(response, (200,))


def ensure_repo(token: str, owner: str, repo_name: str, description: str) -> str:
    repo_url = f"{GITHUB_API}/repos/{owner}/{repo_name}"
    response = github_request("GET", repo_url, token)
    if response.status_code == 200:
        return f"{owner}/{repo_name}"
    if response.status_code != 404:
        require_ok(response, (200,))

    create_response = github_request(
        "POST",
        f"{GITHUB_API}/user/repos",
        token,
        json={
            "name": repo_name,
            "description": description,
            "private": False,
            "auto_init": False,
        },
    )
    data = require_ok(create_response, (201,))
    return data["full_name"]


def get_file_sha(token: str, repo_full_name: str, path_in_repo: str) -> str | None:
    response = github_request(
        "GET",
        f"{GITHUB_API}/repos/{repo_full_name}/contents/{path_in_repo}",
        token,
    )
    if response.status_code == 404:
        return None
    data = require_ok(response, (200,))
    return data.get("sha")


def upload_file(token: str, repo_full_name: str, path_in_repo: str, file_path: Path) -> None:
    content = base64.b64encode(file_path.read_bytes()).decode("ascii")
    payload = {
        "message": f"Add/update {path_in_repo}",
        "content": content,
    }
    current_sha = get_file_sha(token, repo_full_name, path_in_repo)
    if current_sha:
        payload["sha"] = current_sha

    response = github_request(
        "PUT",
        f"{GITHUB_API}/repos/{repo_full_name}/contents/{path_in_repo}",
        token,
        json=payload,
    )
    require_ok(response, (200, 201))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Создать публичный GitHub-репозиторий и загрузить туда проект."
    )
    parser.add_argument(
        "--github-token",
        default=os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
        help="GitHub token с правом создания/записи в repo",
    )
    parser.add_argument(
        "--repo-name",
        default="telegram-gemini-bot",
        help="Имя репозитория на GitHub",
    )
    parser.add_argument(
        "--owner",
        default="",
        help="GitHub username или org. Если не указан, используется владелец токена.",
    )
    args = parser.parse_args()

    token = (args.github_token or "").strip()
    if not token:
        raise SystemExit("Не передан GitHub token.")

    user = get_authenticated_user(token)
    owner = (args.owner or user["login"]).strip()
    repo_full_name = ensure_repo(
        token=token,
        owner=owner,
        repo_name=args.repo_name,
        description="Telegram bot with Gemini integration for Render deployment",
    )

    files = iter_files(PROJECT_DIR)
    for file_path in files:
        relative = file_path.relative_to(PROJECT_DIR).as_posix()
        upload_file(token, repo_full_name, relative, file_path)
        print(f"Uploaded: {relative}")

    print(f"Repository ready: https://github.com/{repo_full_name}")


if __name__ == "__main__":
    main()
