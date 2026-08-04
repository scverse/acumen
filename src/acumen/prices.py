"""Per-model token rates, and the cost of one run computed from its token counts.

The provider-reported dollar figure is not a usable benchmark metric: the Claude SDK
returns one, ``codex exec`` returns none at all, so a mixed matrix would compare a real
number against a zero. Tokens, by contrast, are reported by both. So acumen records the
token breakdown as the primary measurement and derives cost from a rate table here —
one arithmetic path, identical for every provider, and reproducible after the fact.

Two properties this design depends on:

**The breakdown must survive.** Cached input is billed at a tenth of the base rate and
cache writes at 1.25x, so collapsing reads, writes, and fresh input into a single
``input_tokens`` figure and multiplying by the base rate overcharges a cached run by up
to 10x. Benchmark agents share a large prompt prefix, so most of their input *is* cache
reads — the breakdown is the difference between a right answer and a wrong one.

**Rates are frozen into the run.** Prices change; a benchmark comparing a January run
against a July one must not silently re-price the old one. :func:`price_run` is called
once, when the run finishes, and the rates it used are written into ``result.json``
beside the figure they produced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

#: The date :data:`DEFAULT_RATES` was last verified against each provider's own pricing
#: page. Shown in the report so a stale table is visible rather than assumed current.
RATES_AS_OF = "2026-08-03"

#: Where the numbers came from, for the next person who has to re-verify them.
RATE_SOURCES = (
    "https://platform.claude.com/docs/en/pricing",
    "https://developers.openai.com/api/docs/pricing",
)


@dataclass(frozen=True)
class Rates:
    """What one model costs, in USD per million tokens.

    ``cached_input`` and ``cache_write`` are absolute rates, not multipliers — both
    providers happen to price them at 0.1x and 1.25x of ``input``, but that is their
    convention to change, not an invariant worth encoding.
    """

    input: float
    cached_input: float
    cache_write: float
    output: float

    @classmethod
    def from_mapping(cls, value: Any, *, model: str) -> Rates:
        """Build rates from a config mapping, rejecting anything unusable.

        Only ``input`` and ``output`` are required: a provider that does not price
        caching separately (or a model a user is pricing by hand) can omit the cache
        rates, which then default to the standard 0.1x / 1.25x of ``input``.
        """
        if not isinstance(value, dict):
            raise ValueError(f"prices for {model!r} must be a mapping, got {type(value).__name__}")
        unknown = set(value) - {"input", "cached_input", "cache_write", "output"}
        if unknown:
            raise ValueError(f"prices for {model!r} has unknown keys: {sorted(unknown)}")
        for key in ("input", "output"):
            if key not in value:
                raise ValueError(f"prices for {model!r} is missing required key {key!r}")
        numbers: dict[str, float] = {}
        for key, raw in value.items():
            if isinstance(raw, bool) or not isinstance(raw, int | float) or raw < 0:
                raise ValueError(f"prices for {model!r}: {key!r} must be a non-negative number, got {raw!r}")
            numbers[key] = float(raw)
        return cls(
            input=numbers["input"],
            cached_input=numbers.get("cached_input", round(numbers["input"] * 0.1, _CENTS)),
            cache_write=numbers.get("cache_write", round(numbers["input"] * 1.25, _CENTS)),
            output=numbers["output"],
        )


#: Derived cache rates are rounded to this many decimals. Without it ``0.20 * 0.1`` is
#: ``0.020000000000000004``, which never equals the ``$0.02`` a provider publishes — so
#: every price refresh would report a change that isn't one.
_CENTS = 6


def _rates(input_: float, output: float, *, cached: float | None = None, write: float | None = None) -> Rates:
    """Both providers price cache reads at 0.1x input and writes at 1.25x."""
    return Rates(
        input=input_,
        cached_input=round(input_ * 0.1, _CENTS) if cached is None else cached,
        cache_write=round(input_ * 1.25, _CENTS) if write is None else write,
        output=output,
    )


#: USD per million tokens, verified :data:`RATES_AS_OF` against :data:`RATE_SOURCES`.
#: Override or extend from ``config.yaml`` (``prices:``) rather than editing this — a
#: model missing here is recorded unpriced, never guessed at.
DEFAULT_RATES: dict[str, Rates] = {
    # Anthropic. Sonnet 5 carries an introductory rate through 2026-08-31; the standard
    # $3/$15 resumes after it, so re-verify this row rather than trusting it past then.
    "claude-opus-5": _rates(5.00, 25.00),
    "claude-opus-4-8": _rates(5.00, 25.00),
    "claude-opus-4-7": _rates(5.00, 25.00),
    "claude-sonnet-5": _rates(2.00, 10.00),
    "claude-sonnet-4-6": _rates(3.00, 15.00),
    "claude-haiku-4-5": _rates(1.00, 5.00),
    "claude-haiku-4-5-20251001": _rates(1.00, 5.00),
    # OpenAI.
    "gpt-5.6-sol": _rates(5.00, 30.00),
    "gpt-5.6-terra": _rates(2.00, 12.00),
    "gpt-5.6-luna": _rates(0.20, 1.20),
}


def resolve_rates(model: str, overrides: dict[str, Rates] | None = None) -> Rates | None:
    """Find the rates for ``model``, or ``None`` if it is not priced.

    Config overrides win over the built-in table, so an operator on negotiated rates or a
    gateway can price their own traffic. A provider-qualified ID (``openai/gpt-5.6-sol``)
    falls back to its bare form, which keeps proxy configurations priceable.
    """
    for table in (overrides or {}, DEFAULT_RATES):
        for key in (model, model.strip().lower(), model.rsplit("/", 1)[-1].strip().lower()):
            if key in table:
                return table[key]
    return None


@dataclass(frozen=True)
class Usage:
    """One run's token counts, normalized across providers.

    ``input`` is the *total* input including both cached classes — the shape both
    providers report — so the fresh-token count is ``input - cache_read - cache_write``.
    """

    input: int
    cache_read: int
    cache_write: int
    output: int

    @property
    def total(self) -> int:
        """Every token the run consumed, input and output together."""
        return self.input + self.output

    @property
    def fresh_input(self) -> int:
        """Input billed at the full rate.

        Clamped at zero: a provider is free to redefine how the parts relate, and a
        negative token count would silently become a negative cost.
        """
        return max(0, self.input - self.cache_read - self.cache_write)


def _int(usage: dict, key: str) -> int:
    return int(usage.get(key, 0) or 0)


def normalize_usage(usage: dict | None, *, provider: str = "claude") -> Usage:
    """Normalize a provider's usage block, keeping the cache split intact.

    The two providers report the same information in different shapes. Codex's
    ``input_tokens`` is the total with ``cached_input_tokens`` as a subset of it; the
    Claude SDK reports the three classes side by side, so the total is their sum. Both
    are normalized to :class:`Usage`, whose ``input`` is always the total.

    The split is kept rather than collapsed because the classes are priced an order of
    magnitude apart — see this module's docstring. Verified against a live ``turn.completed``
    event, whose usage block is::

        {
            "input_tokens": 12051,
            "cached_input_tokens": 8960,
            "cache_write_input_tokens": 0,
            "output_tokens": 5,
            "reasoning_output_tokens": 0,
        }

    ``reasoning_output_tokens`` is a subset of ``output_tokens`` and billed as output, so
    the total is the figure to use and the subset is ignored.
    """
    if not usage:
        return Usage(input=0, cache_read=0, cache_write=0, output=0)
    output = _int(usage, "output_tokens")
    if provider == "codex":
        return Usage(
            input=_int(usage, "input_tokens"),
            cache_read=_int(usage, "cached_input_tokens"),
            cache_write=_int(usage, "cache_write_input_tokens"),
            output=output,
        )
    cache_read = _int(usage, "cache_read_input_tokens")
    cache_write = _int(usage, "cache_creation_input_tokens")
    return Usage(
        input=_int(usage, "input_tokens") + cache_read + cache_write,
        cache_read=cache_read,
        cache_write=cache_write,
        output=output,
    )


def pricer(model: str, overrides: dict[str, Rates] | None = None) -> Callable[[dict], float | None]:
    """A usage→USD hook for a provider that reports no billed figure of its own.

    Codex cannot enforce a budget cap without one: it reports tokens, so the cap has to be
    applied to the same rate table the run is priced by, never a second one — otherwise a run
    could be stopped at a budget it is not billed against. Returns ``None`` for an unpriced
    model, which callers must read as "no cap can be enforced", never as "free".
    """
    rates = resolve_rates(model, overrides)
    return lambda usage: price_run(normalize_usage(usage, provider="codex"), rates)


def price_run(usage: Usage, rates: Rates | None) -> float | None:
    """Cost of one run in USD, or ``None`` when the model has no rates.

    ``None`` is deliberately distinct from ``0.0``: an unpriced model must not be
    reported as a free one, which is exactly the failure the provider-reported figure
    produced for Codex.
    """
    if rates is None:
        return None
    per_token = 1_000_000
    return (
        usage.fresh_input * rates.input
        + usage.cache_read * rates.cached_input
        + usage.cache_write * rates.cache_write
        + usage.output * rates.output
    ) / per_token


def rates_payload(rates: Rates | None) -> dict[str, float] | None:
    """The rates as they go into ``result.json`` — the run's frozen price provenance."""
    return None if rates is None else asdict(rates)
