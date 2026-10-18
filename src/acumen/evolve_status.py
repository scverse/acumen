"""One-screen health check for a running experiment: progress, spend, and the failure modes we paid for.

    acumen status            # looks under ./runs
    acumen status runs_sanity

Prints per-journal generation counts and decisions, then three health checks. Exit code 1 when a
check fails, so a polling agent can branch on it without parsing the text.
"""

from __future__ import annotations

import json
from pathlib import Path

from acumen.grade import INVALID_REASONS
from acumen.transient import is_transient


def journals(root: Path) -> list[Path]:
    """Every evolve.jsonl under the run tree: the main one and one per island."""
    return sorted(root.rglob("evolve.jsonl"))


def main(argv: list[str]) -> int:
    """Print progress and health for the run tree named in ``argv`` (default ``runs``)."""
    root = Path(argv[0] if argv else "runs")
    if not root.is_dir():
        print(f"no run tree at {root}")
        return 2
    problems: list[str] = []

    for path in journals(root):
        gens = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not gens:
            continue
        accepted = sum(g["accepted"] for g in gens)
        promoted = sum(1 for g in gens if g["confirm_promoted"] is True)
        reverted = sum(1 for g in gens if g["confirm_promoted"] is False)
        last = gens[-1]
        cost = sum(g["cost_usd"] for g in gens)
        print(
            f"{path.parent.relative_to(root.parent)}: {len(gens)} generations, {accepted} accepted, "
            f"confirmations {promoted} promoted / {reverted} reverted, champion {last['confirmed']}, ${cost:,.0f}"
        )
        c, k = last["champion_screen"], last["candidate_screen"]
        verdict = "ACCEPT" if last["accepted"] else "reject"
        print(f"  last: gen {last['generation']} {last['candidate']} {k[0]}/{k[1]} vs {c[0]}/{c[1]} -> {verdict}")
        # Selection health: accepting most candidates means the bar is inside the noise; accepting
        # none for a long stretch means it is above anything the rulebook edits can reach.
        if len(gens) >= 6 and accepted / len(gens) > 0.5:
            problems.append(f"{path}: {accepted}/{len(gens)} accepted — --accept-delta is likely inside the noise")
        if len(gens) >= 10 and accepted == 0:
            problems.append(f"{path}: 0 of {len(gens)} accepted — --accept-delta may be unreachably high")

    runs = pending_invalid = mislabelled = errored = loaded = skill_runs = 0
    spend = 0.0
    for f in root.rglob("result.json"):
        r = json.loads(f.read_text())
        runs += 1
        spend += r.get("cost_usd") or 0.0
        if r.get("valid") is False or r.get("reason") in INVALID_REASONS:
            # Recorded as infrastructure-invalid: the cell stays pending and the identical rerun
            # replaces it. Expected after a pause (exit 3); a problem only if it never clears.
            pending_invalid += 1
        elif r.get("reason") == "error":
            if is_transient(r.get("error") or ""):
                mislabelled += 1
            else:
                errored += 1  # turn cap, SDK buffer overflow: a real (if unlucky) failure of the run
        if r.get("skill_name"):
            skill_runs += 1
            loaded += bool(r.get("skill_loaded"))
    load = f"{loaded}/{skill_runs} ({loaded / skill_runs:.0%})" if skill_runs else "n/a"
    print(
        f"bench runs: {runs}, spend ${spend:,.0f}, skill loaded in {load}, "
        f"invalid-and-pending {pending_invalid}, platform-killed-but-recorded {mislabelled}, "
        f"other errors {errored} (turn cap etc. — counted as failures)"
    )
    # A platform failure is supposed to be recorded invalid (valid: false) and rerun. One that was
    # recorded as an ordinary task failure is silently dragging an arm down.
    if mislabelled:
        problems.append(
            f"{mislabelled} platform-killed run(s) recorded as task failures — delete those result.json and rerun"
        )
    if skill_runs >= 30 and loaded / skill_runs < 0.8:
        problems.append(f"skill loaded in only {load} of skill runs — a draft may not be triggering at all")

    for p in problems:
        print(f"PROBLEM: {p}")
    if not problems:
        print("health: ok")
    return 1 if problems else 0
