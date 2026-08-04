"""Refresh :data:`acumen.prices.DEFAULT_RATES` from the providers' own pricing pages.

Rates drift — OpenAI cut one tier by 80% in July 2026, and Claude Sonnet 5's introductory
rate expires 2026-08-31 — so a table baked into a release goes stale on its own. This
module fetches both providers' published prices and reports what changed.

**This runs on request, never during a benchmark.** ``acumen bench`` reads the frozen
table and touches the network only to run agents. Three reasons that separation is
deliberate rather than timid:

- A benchmark that re-prices itself from a live page is not reproducible, and one that
  fails because a docs site moved is worse than one using a table labelled with its age.
- Parsing has to choose a tier (standard, not batch/flex/fast), a context band (short —
  the long-context band is priced ~2x and its threshold is not stated per model), and,
  for dated rows, which side of the date we are on. Each is a place a silent mis-parse
  yields a plausible wrong number, and cost is a headline metric of the report.
- So the output is a **diff for a human to accept**, not an overwrite.

Both providers publish Markdown alongside their HTML, which is what makes this tractable:
these are documented tables, not scraped page furniture.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

from acumen.prices import DEFAULT_RATES, Rates

#: The Markdown pricing pages. Both are the ``.md`` twin of the human docs page — append
#: ``.md`` to a docs URL on either site.
PRICE_SOURCES = {
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing.md",
    "openai": "https://developers.openai.com/api/docs/pricing.md",
}

#: What the parsed rates mean. Recorded alongside them so the choice is never implicit.
PRICE_TIER = "standard, short-context"


class PriceFeedError(RuntimeError):
    """Raised when a pricing page cannot be fetched or yields no usable rows."""


def fetch(url: str, *, timeout: float = 30.0) -> str:
    """Fetch one pricing page as text."""
    request = urllib.request.Request(url, headers={"User-Agent": "acumen-pricefeed"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise PriceFeedError(f"could not fetch {url}: {err}") from err


def _cells(line: str) -> list[str]:
    """Split one Markdown table row into trimmed cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _tables(markdown: str) -> list[tuple[list[str], list[list[str]]]]:
    """Return every Markdown table as ``(header cells, body rows)``."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("|") or index + 1 >= len(lines):
            continue
        separator = lines[index + 1].strip()
        if not re.fullmatch(r"\|[\s:|-]+\|", separator):
            continue
        header = _cells(line)
        rows = []
        for row in lines[index + 2 :]:
            if not row.lstrip().startswith("|"):
                break
            rows.append(_cells(row))
        tables.append((header, rows))
    return tables


def _money(cell: str) -> float | None:
    """Parse a price cell (``$6.25 / MTok``, ``$0.02``, ``-``) to a number."""
    match = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", cell)
    return float(match.group(1)) if match else None


_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _model_label(cell: str) -> str:
    """Strip Markdown links and parenthetical qualifiers from a model cell."""
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", _LINK.sub(r"\1", cell))).strip()


def _slug(label: str) -> str:
    """Turn a display name into its API model ID (``Claude Haiku 4.5`` → ``claude-haiku-4-5``)."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", label.lower())).strip("-")


_THROUGH = re.compile(r"\bthrough\s+([A-Z][a-z]+ \d{1,2}, \d{4})")
_STARTING = re.compile(r"\bstarting\s+([A-Z][a-z]+ \d{1,2}, \d{4})")


def _dated_window(label: str, today: date) -> tuple[str, bool]:
    """Split a dated row label into its model name and whether it applies on ``today``.

    Anthropic prices a promotional rate as two rows — "… through August 31, 2026" and
    "… starting September 1, 2026". Taking whichever row parses first would silently
    pick up a rate that is not in effect.
    """
    applies = True
    for pattern, active in ((_THROUGH, lambda d: today <= d), (_STARTING, lambda d: today >= d)):
        match = pattern.search(label)
        if match is None:
            continue
        boundary = datetime.strptime(match.group(1), "%B %d, %Y").date()
        applies = active(boundary)
        label = label[: match.start()].strip()
    return label, applies


def parse_anthropic(markdown: str, *, today: date) -> dict[str, Rates]:
    """Parse Anthropic's per-model table (base input / cache writes / cache hits / output)."""
    rates: dict[str, Rates] = {}
    for header, rows in _tables(markdown):
        lowered = [cell.lower() for cell in header]
        if not (any("base input" in c for c in lowered) and any("cache hits" in c for c in lowered)):
            continue
        columns = {
            "input": next(i for i, c in enumerate(lowered) if "base input" in c),
            "cache_write": next(i for i, c in enumerate(lowered) if "5m cache" in c),
            "cache_write_5m": next(i for i, c in enumerate(lowered) if "5m cache" in c),
            "cache_write_1h": next(i for i, c in enumerate(lowered) if "1h cache" in c),
            "cached_input": next(i for i, c in enumerate(lowered) if "cache hits" in c),
            "output": next(i for i, c in enumerate(lowered) if "output" in c),
        }
        for row in rows:
            if len(row) <= max(columns.values()):
                continue
            label, applies = _dated_window(_model_label(row[0]), today)
            if not applies or not label.lower().startswith("claude"):
                continue
            values = {name: _money(row[index]) for name, index in columns.items()}
            if any(value is None for value in values.values()):
                continue
            rates[_slug(label)] = Rates(**values)  # type: ignore[arg-type]
    if not rates:
        raise PriceFeedError("no Claude rates found — the pricing page layout may have changed")
    return rates


def parse_openai(markdown: str) -> dict[str, Rates]:
    """Parse OpenAI's standard-tier table, short-context columns only."""
    rates: dict[str, Rates] = {}
    for header, rows in _tables(markdown):
        lowered = [cell.lower() for cell in header]
        wanted = {
            "input": "short context input",
            "cached_input": "short context cached input",
            "cache_write": "short context cache writes",
            "output": "short context output",
        }
        if not all(any(c == want for c in lowered) for want in wanted.values()):
            continue
        columns = {name: lowered.index(want) for name, want in wanted.items()}
        for row in rows:
            if len(row) <= max(columns.values()):
                continue
            model = _model_label(row[0])
            if not model or " " in model:
                continue
            values = {name: _money(row[index]) for name, index in columns.items()}
            if values["input"] is None or values["output"] is None:
                continue
            # A model can omit cache columns ("-"); fall back to the standard multiples.
            rates.setdefault(
                model.lower(),
                Rates(
                    input=values["input"],
                    cached_input=values["cached_input"] or values["input"] * 0.1,
                    cache_write=values["cache_write"] or values["input"] * 1.25,
                    output=values["output"],
                ),
            )
        # The standard table comes first; batch/flex/fast repeat the same headers.
        break
    if not rates:
        raise PriceFeedError("no OpenAI rates found — the pricing page layout may have changed")
    return rates


@dataclass(frozen=True)
class RateChange:
    """One model's rates differing between the shipped table and the published page."""

    model: str
    before: Rates | None
    after: Rates

    @property
    def kind(self) -> str:
        """Whether the shipped table lacks this model entirely, or prices it differently."""
        return "new" if self.before is None else "changed"

    def describe(self) -> str:
        """One reviewable line: what moved, from what, to what."""
        if self.before is None:
            return f"  + {self.model}: in ${self.after.input} out ${self.after.output} (new)"
        # Name every field that moved — a cache rate can change while input/output hold,
        # and on a cached workload that is most of the bill.
        moved = [
            f"{field} ${getattr(self.before, field)}→${getattr(self.after, field)}"
            for field in (
                "input",
                "cached_input",
                "cache_write",
                "cache_write_5m",
                "cache_write_1h",
                "output",
            )
            if getattr(self.before, field) != getattr(self.after, field)
        ]
        return f"  ~ {self.model}: {', '.join(moved)}"


def refresh(*, today: date, fetcher=fetch) -> dict[str, Rates]:
    """Fetch and parse both providers' published rates."""
    markdown = {name: fetcher(url) for name, url in PRICE_SOURCES.items()}
    return {**parse_anthropic(markdown["anthropic"], today=today), **parse_openai(markdown["openai"])}


def diff_rates(current: dict[str, Rates], fetched: dict[str, Rates]) -> list[RateChange]:
    """Models whose published rates differ from ``current``, plus ones it doesn't cover.

    Only models present in the fetch are reported: a model missing from the page has been
    retired or renamed, which is not something to silently drop from a working table.
    """
    changes = [
        RateChange(model=model, before=current.get(model), after=rates)
        for model, rates in sorted(fetched.items())
        if current.get(model) != rates
    ]
    return changes


def to_yaml_block(rates: dict[str, Rates]) -> str:
    """Render rates as a ``prices:`` block for ``config.yaml``."""
    lines = ["prices:"]
    for model, rate in sorted(rates.items()):
        lines.append(f"  {model}:")
        lines.append(f"    input: {rate.input}")
        lines.append(f"    cached_input: {rate.cached_input}")
        lines.append(f"    cache_write: {rate.cache_write}")
        if rate.cache_write_5m is not None:
            lines.append(f"    cache_write_5m: {rate.cache_write_5m}")
        if rate.cache_write_1h is not None:
            lines.append(f"    cache_write_1h: {rate.cache_write_1h}")
        lines.append(f"    output: {rate.output}")
    return "\n".join(lines) + "\n"


def shipped_rates() -> dict[str, Rates]:
    """The table a refresh is compared against."""
    return dict(DEFAULT_RATES)
