"""Live Codabench submission driver for the NovelQA leaderboard.

The platform's REST flow (extracted from a logged-in browser
session against codabench.org/competitions/2727):

    1. Authenticate.
       Two paths:
         (a) cookie-based — paste `sessionid` + `csrftoken` from a
             logged-in browser session into the env (preferred; no
             password stored).
         (b) username + password — script does the Django allauth
             login dance, fetching the CSRF token from the login
             page and POSTing credentials.

    2. POST /api/datasets/  with type=submission, competition=2727,
       file_name=<zip name>, file_size=<bytes>.
       Response includes a UUID + a signed S3 URL.

    3. PUT  the zip to the signed S3 URL with content-type
       "application/zip".

    4. PUT  /api/datasets/completed/<uuid>/  to finalise the upload.

    5. POST /api/submissions/  with data=<uuid>, phase=4713,
       tasks=[6466].
       Response includes the new submission id.

    6. GET  /api/submissions/<id>/  in a poll loop until status
       reaches "Finished" / "Failed".

    7. GET  /api/phases/4713/get_leaderboard/  to read back the
       posted accuracy.

The phase id (4713) and task id (6466) are NovelQA-competition
specific; both are recorded as constants below and verified on
first request via /api/competitions/2727/.

Authentication notes
--------------------
Codabench's API uses Django's session + CSRF cookies. CSRF token
must be sent as both a cookie and an `X-CSRFToken` header on
state-changing requests (POST/PUT). The CSRF cookie is named
``csrftoken``; the session cookie is named ``sessionid``.

Environment variables read (any one auth path is enough)::

    # Path A — pre-existing cookies (preferred)
    CODABENCH_SESSIONID=<value>
    CODABENCH_CSRFTOKEN=<value>

    # Path B — username + password (convenient but stores password)
    CODABENCH_USERNAME=<value>
    CODABENCH_PASSWORD=<value>

If both are set, Path A is tried first and Path B is the fallback.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from pilot.env import load_env


_BASE_URL = "https://www.codabench.org"
# NovelQA competition. Constants are extracted from the live
# competition page; see module docstring for the cross-reference.
_NOVELQA_COMPETITION_ID = 2727
_NOVELQA_PHASE_ID = 4713
_NOVELQA_TASK_ID = 6466


@dataclass
class SubmissionResult:
    submission_id: int | None
    status: str
    leaderboard_rows: list[dict[str, Any]]
    raw_submission: dict[str, Any]


# ──────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────

def _login_with_password(
    client: httpx.Client, username: str, password: str
) -> None:
    """Drive Django allauth's login form. Sets cookies on the client."""
    # 1. GET the login page to obtain CSRF cookie + a hidden form token.
    response = client.get("/accounts/login/")
    response.raise_for_status()
    csrf_match = re.search(
        r'name=["\']csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)["\']',
        response.text,
    )
    if not csrf_match:
        raise RuntimeError(
            "Could not find csrfmiddlewaretoken on the login page. "
            "The Codabench login form layout may have changed."
        )
    csrf_form_token = csrf_match.group(1)

    # 2. POST credentials.
    response = client.post(
        "/accounts/login/",
        data={
            "login": username,
            "password": password,
            "csrfmiddlewaretoken": csrf_form_token,
        },
        headers={"Referer": f"{_BASE_URL}/accounts/login/"},
        follow_redirects=False,
    )
    # 302 to /profile or similar is a successful login; 200 with the
    # login form re-rendered means failure.
    if response.status_code in (301, 302):
        return
    if response.status_code == 200 and "id_login" in response.text:
        raise RuntimeError(
            "Codabench rejected the credentials. Check CODABENCH_USERNAME "
            "and CODABENCH_PASSWORD in .env."
        )
    response.raise_for_status()


def _build_authenticated_client() -> httpx.Client:
    """Return an httpx.Client with valid session cookies set."""
    sessionid = os.environ.get("CODABENCH_SESSIONID")
    csrftoken = os.environ.get("CODABENCH_CSRFTOKEN")

    client = httpx.Client(
        base_url=_BASE_URL,
        timeout=120.0,
        follow_redirects=True,
        headers={
            "User-Agent": "msc-thesis-pilot/0.1 (CodabenchSubmitter)",
            "Referer": f"{_BASE_URL}/competitions/{_NOVELQA_COMPETITION_ID}/",
        },
    )

    if sessionid and csrftoken:
        # Path A: paste cookies directly.
        client.cookies.set("sessionid", sessionid, domain="www.codabench.org")
        client.cookies.set("csrftoken", csrftoken, domain="www.codabench.org")
        return client

    username = os.environ.get("CODABENCH_USERNAME")
    password = os.environ.get("CODABENCH_PASSWORD")
    if username and password:
        # Path B: full login dance.
        _login_with_password(client, username, password)
        if "sessionid" not in client.cookies:
            raise RuntimeError(
                "Login did not produce a sessionid cookie; the response "
                "may have indicated 2FA or a redirect we did not follow."
            )
        return client

    raise RuntimeError(
        "No Codabench credentials in environment. Set either "
        "CODABENCH_SESSIONID + CODABENCH_CSRFTOKEN (preferred) or "
        "CODABENCH_USERNAME + CODABENCH_PASSWORD."
    )


def _csrf_headers(client: httpx.Client) -> dict[str, str]:
    """X-CSRFToken header required for POST/PUT to /api/."""
    token = client.cookies.get("csrftoken")
    if not token:
        raise RuntimeError("No csrftoken cookie present after auth.")
    return {"X-CSRFToken": token}


# ──────────────────────────────────────────────────────────────────────
# Submission flow
# ──────────────────────────────────────────────────────────────────────

def _initiate_dataset(
    client: httpx.Client, *, zip_path: Path
) -> dict[str, Any]:
    """Step 2: ask Codabench to allocate a submission slot + signed S3 URL."""
    payload = {
        "type": "submission",
        "competition": _NOVELQA_COMPETITION_ID,
        "request_sassy_file_name": zip_path.name,
        "file_name": zip_path.name,
        "file_size": zip_path.stat().st_size,
    }
    response = client.post(
        "/api/datasets/",
        json=payload,
        headers=_csrf_headers(client),
    )
    response.raise_for_status()
    return response.json()


def _upload_to_s3(signed_url: str, zip_path: Path) -> None:
    """Step 3: PUT the zip to the signed MinIO/S3 URL."""
    with httpx.Client(timeout=300.0) as raw_client:
        with zip_path.open("rb") as fh:
            response = raw_client.put(
                signed_url,
                content=fh.read(),
                headers={"Content-Type": "application/zip"},
            )
        response.raise_for_status()


def _mark_complete(client: httpx.Client, dataset_uuid: str) -> dict[str, Any]:
    """Step 4: tell Codabench the upload finished."""
    response = client.put(
        f"/api/datasets/completed/{dataset_uuid}/",
        json={},
        headers=_csrf_headers(client),
    )
    response.raise_for_status()
    return response.json() if response.text else {}


def _create_submission(
    client: httpx.Client, dataset_uuid: str
) -> dict[str, Any]:
    """Step 5: register the submission against the NovelQA phase + task."""
    payload = {
        "data": dataset_uuid,
        "phase": _NOVELQA_PHASE_ID,
        "fact_sheet_answers": None,
        "tasks": [_NOVELQA_TASK_ID],
        "organization": None,
        "queue": None,
    }
    response = client.post(
        "/api/submissions/",
        json=payload,
        headers=_csrf_headers(client),
    )
    response.raise_for_status()
    return response.json()


def _poll_submission(
    client: httpx.Client,
    submission_id: int,
    *,
    poll_interval_s: float = 10.0,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Step 6: GET the submission until it reaches a terminal status."""
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/submissions/{submission_id}/")
        response.raise_for_status()
        last = response.json()
        status = (last.get("status") or "").lower()
        if status in {"finished", "failed", "cancelled", "rejected"}:
            return last
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Submission {submission_id} did not finish within {timeout_s}s; "
        f"last status: {last.get('status')!r}"
    )


def _fetch_leaderboard(client: httpx.Client) -> list[dict[str, Any]]:
    """Step 7: pull the leaderboard for the NovelQA phase."""
    response = client.get(
        f"/api/phases/{_NOVELQA_PHASE_ID}/get_leaderboard/",
        params={"page": 1, "page_size": 50},
    )
    response.raise_for_status()
    body = response.json()
    if isinstance(body, dict) and "results" in body:
        return list(body["results"])
    return list(body) if isinstance(body, list) else []


# ──────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────

def submit_zip(
    zip_path: Path,
    *,
    poll_interval_s: float = 10.0,
    timeout_s: float = 1800.0,
) -> SubmissionResult:
    """Drive the full upload + submit + poll + leaderboard flow.

    Args:
        zip_path: path to a Codabench-ready submission.zip (built via
            ``pilot.codabench.format.write_submission_zip``).
        poll_interval_s: seconds between submission status checks.
        timeout_s: hard deadline for terminal submission status.

    Returns:
        SubmissionResult with the submission id, terminal status,
        leaderboard rows, and the raw submission record.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"missing submission zip: {zip_path}")

    with _build_authenticated_client() as client:
        dataset = _initiate_dataset(client, zip_path=zip_path)
        signed_url = dataset.get("file_url") or dataset.get("upload_url") or dataset.get("sassy_url")
        if not signed_url:
            raise RuntimeError(
                f"Dataset response missing signed-URL field; got keys: "
                f"{sorted(dataset.keys())}"
            )
        dataset_uuid = dataset.get("uuid") or dataset.get("key") or dataset.get("id")
        if not dataset_uuid:
            raise RuntimeError(
                f"Dataset response missing uuid/id field; got keys: "
                f"{sorted(dataset.keys())}"
            )

        _upload_to_s3(signed_url, zip_path)
        _mark_complete(client, dataset_uuid)
        submission = _create_submission(client, dataset_uuid)
        sub_id = submission.get("id")
        if sub_id is None:
            raise RuntimeError(
                f"Create-submission response missing id; got: {submission}"
            )

        terminal = _poll_submission(
            client, sub_id, poll_interval_s=poll_interval_s, timeout_s=timeout_s
        )
        leaderboard = _fetch_leaderboard(client)

    return SubmissionResult(
        submission_id=sub_id,
        status=str(terminal.get("status", "")),
        leaderboard_rows=leaderboard,
        raw_submission=terminal,
    )


def main() -> int:
    import argparse

    load_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--poll-interval-s", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    try:
        result = submit_zip(
            zip_path=args.zip,
            poll_interval_s=args.poll_interval_s,
            timeout_s=args.timeout_s,
        )
    except Exception as exc:
        print(f"submission failed: {exc!r}", file=sys.stderr)
        return 1

    print(json.dumps({
        "submission_id": result.submission_id,
        "status": result.status,
        "leaderboard_rows": result.leaderboard_rows,
    }, indent=2))
    return 0 if result.status.lower() == "finished" else 2


if __name__ == "__main__":
    sys.exit(main())
