"""Per-model token rates and provider-first cost resolution.

The Claude SDK reports an API-equivalent dollar cost while ``codex exec`` reports none.
Acumen therefore preserves two measurements: the provider value when one exists and a
reproducible inference from a rate table. The provider value is canonical; inference is
the fallback and remains alongside it for comparison.

Three properties this design depends on:

**The breakdown must survive.** Cached input is billed at a tenth of the base rate and
cache writes at 1.25x, so collapsing reads, writes, and fresh input into a single
``input_tokens`` figure and multiplying by the base rate overcharges a cached run by up
to 10x. Benchmark agents share a large prompt prefix, so most of their input *is* cache
reads — the breakdown is the difference between a right answer and a wrong one.

**Rates are frozen into the run.** Prices change; a benchmark comparing a January run
against a July one must not silently re-price the old one. :func:`price_run` is called
once, when the run finishes, and the rates it used are written into ``result.json``
beside the figure they produced.

**What was frozen must have been right when it froze.** Freezing is only worth doing if
the rates were current at the time, so this module ships **no rate table of its own**.
Rates come from the providers' live pricing pages (see :mod:`acumen.pricefeed`) or from
the operator's ``prices:`` config, assembled into a :class:`PriceTable`. A baked-in table
would only ever be right on the day it was written, and a wrong rate that has been frozen
into a result is indistinguishable from a right one unless the provenance travels with it
— so each run records where its rates came from and how old they were.

A model no layer prices is recorded **unpriced**, which is a first-class state throughout:
``cost_usd`` is ``None``, never ``0.0``, and reports show it as unknown rather than free.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rates:
    """What one model costs, in USD per million tokens.

    All cache fields are absolute rates, not multipliers. ``cache_write`` is the legacy
    aggregate rate; the optional duration-specific fields retain Claude's distinct
    five-minute and one-hour write prices.
    """

    input: float
    cached_input: float
    cache_write: float
    output: float
    cache_write_5m: float | None = None
    cache_write_1h: float | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, model: str) -> Rates:
        """Build rates from a config mapping, rejecting anything unusable.

        Only ``input`` and ``output`` are required: a provider that does not price
        caching separately (or a model a user is pricing by hand) can omit the cache
        rates, which then default to the standard 0.1x / 1.25x of ``input``.
        """
        if not isinstance(value, dict):
            raise ValueError(f"prices for {model!r} must be a mapping, got {type(value).__name__}")
        unknown = set(value) - {
            "input",
            "cached_input",
            "cache_write",
            "cache_write_5m",
            "cache_write_1h",
            "output",
        }
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
        cache_write = numbers.get("cache_write", round(numbers["input"] * 1.25, _CENTS))
        return cls(
            input=numbers["input"],
            cached_input=numbers.get("cached_input", round(numbers["input"] * 0.1, _CENTS)),
            cache_write=cache_write,
            output=numbers["output"],
            cache_write_5m=numbers.get("cache_write_5m", cache_write),
            cache_write_1h=numbers.get("cache_write_1h", round(numbers["input"] * 2, _CENTS)),
        )


#: Derived cache rates are rounded to this many decimals. Without it ``0.20 * 0.1`` is
#: ``0.020000000000000004``, which never equals the ``$0.02`` a provider publishes — so
#: every price refresh would report a change that isn't one.
_CENTS = 6


#: A trailing dated-snapshot suffix, e.g. the ``-20251001`` of ``claude-haiku-4-5-20251001``.
_SNAPSHOT = re.compile(r"-\d{8}$")


def _keys(model: str) -> tuple[str, ...]:
    """The forms ``model`` may be keyed under, most specific first.

    Two normalizations, each covering a way the same model gets named differently:

    - A provider-qualified ID (``openai/gpt-5.6-sol``) falls back to its bare form, which
      keeps proxy and gateway configurations priceable.
    - A dated snapshot (``claude-haiku-4-5-20251001``) falls back to its family
      (``claude-haiku-4-5``). Providers publish one rate per family and pin the snapshot
      for reproducibility, so the pages never list the dated ID that a benchmark pins to.
      Without this, pinning a snapshot — which is the right thing to do — would silently
      leave every run unpriced.
    """
    bare = model.rsplit("/", 1)[-1].strip().lower()
    return tuple(dict.fromkeys((model, model.strip().lower(), bare, _SNAPSHOT.sub("", bare))))


#: Rate provenance, most authoritative first. ``config`` is a deliberate operator
#: statement about their own billing and outranks a published page; ``fetched`` is what
#: the provider published on the day of the run.
CONFIG, FETCHED = "config", "fetched"


@dataclass(frozen=True)
class RateLookup:
    """One model's rates together with where they came from and how current they are.

    ``as_of`` is ``None`` for a ``config`` rate: an operator's own numbers carry no
    verification date, and inventing one would claim currency nothing established.
    """

    rates: Rates
    source: str
    as_of: str | None


@dataclass(frozen=True)
class PriceTable:
    """The rates one command prices its runs by.

    Two layers: ``overrides`` from ``config.yaml``, then ``fetched`` from the providers'
    live pricing pages. Config wins, because only the operator knows what they are
    actually billed — a negotiated rate or a gateway's markup is not on any public page.

    There is deliberately no third layer of built-in defaults. Published prices move, so a
    table compiled into a release is wrong from some unknowable date onward, and because
    each run's cost is frozen at write time that wrongness would be preserved rather than
    corrected on the next run. An empty table prices nothing, which is the honest outcome
    when no rates could be established.
    """

    overrides: dict[str, Rates] = field(default_factory=dict)
    fetched: dict[str, Rates] = field(default_factory=dict)
    fetched_as_of: str | None = None

    def lookup(self, model: str) -> RateLookup | None:
        """Resolve ``model`` through the layers, or ``None`` if no layer prices it."""
        layers: tuple[tuple[str, dict[str, Rates], str | None], ...] = (
            (CONFIG, self.overrides, None),
            (FETCHED, self.fetched, self.fetched_as_of),
        )
        for source, table, as_of in layers:
            for key in _keys(model):
                if key in table:
                    return RateLookup(rates=table[key], source=source, as_of=as_of)
        return None

    def rates(self, model: str) -> Rates | None:
        """Just the rates, for callers that do not record provenance."""
        found = self.lookup(model)
        return None if found is None else found.rates


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
    cache_write_5m: int = 0
    cache_write_1h: int = 0

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
    cache_detail = usage.get("cache_creation")
    if not isinstance(cache_detail, dict):
        cache_detail = {}
    cache_write_5m = _int(cache_detail, "ephemeral_5m_input_tokens") or _int(usage, "cache_creation_5m_input_tokens")
    cache_write_1h = _int(cache_detail, "ephemeral_1h_input_tokens") or _int(usage, "cache_creation_1h_input_tokens")
    cache_write = _int(usage, "cache_creation_input_tokens")
    if cache_write == 0:
        cache_write = cache_write_5m + cache_write_1h
    return Usage(
        input=_int(usage, "input_tokens") + cache_read + cache_write,
        cache_read=cache_read,
        cache_write=cache_write,
        output=output,
        cache_write_5m=cache_write_5m,
        cache_write_1h=cache_write_1h,
    )


def price_usage(
    usage: dict | None,
    *,
    model: str,
    provider: str = "claude",
    prices: PriceTable | None = None,
) -> float | None:
    """Price a provider usage payload through ``prices``, or the built-in table."""
    rates = (prices or PriceTable()).rates(model)
    return price_run(normalize_usage(usage, provider=provider), rates)


def pricer(model: str, prices: PriceTable | None = None) -> Callable[[dict], float | None]:
    """A usage→USD hook for a provider that reports no billed figure of its own.

    Codex cannot enforce a budget cap without one: it reports tokens, so the cap has to be
    applied to the same rate table the run is priced by, never a second one — otherwise a run
    could be stopped at a budget it is not billed against. Returns ``None`` for an unpriced
    model, which callers must read as "no cap can be enforced", never as "free".
    """
    return lambda usage: price_usage(usage, model=model, provider="codex", prices=prices)


def price_run(usage: Usage, rates: Rates | None) -> float | None:
    """Cost of one run in USD, or ``None`` when the model has no rates.

    ``None`` is deliberately distinct from ``0.0``: an unpriced model must not be
    reported as a free one, which is exactly the failure the provider-reported figure
    produced for Codex.
    """
    if rates is None:
        return None
    per_token = 1_000_000
    classified_writes = min(usage.cache_write, usage.cache_write_5m + usage.cache_write_1h)
    aggregate_writes = usage.cache_write - classified_writes
    write_5m_rate = rates.cache_write_5m if rates.cache_write_5m is not None else rates.cache_write
    write_1h_rate = rates.cache_write_1h if rates.cache_write_1h is not None else rates.cache_write
    return (
        usage.fresh_input * rates.input
        + usage.cache_read * rates.cached_input
        + aggregate_writes * rates.cache_write
        + usage.cache_write_5m * write_5m_rate
        + usage.cache_write_1h * write_1h_rate
        + usage.output * rates.output
    ) / per_token


@dataclass(frozen=True)
class CostResolution:
    """Canonical and comparative costs for one completed agent run."""

    cost_usd: float | None
    cost_source: str
    provider_cost_usd: float | None
    inferred_cost_usd: float | None
    cost_delta_usd: float | None
    cost_delta_pct: float | None

    @property
    def available(self) -> bool:
        """Whether this run has any canonical dollar cost."""
        return self.cost_usd is not None


def resolve_cost(provider_cost_usd: float | None, inferred_cost_usd: float | None) -> CostResolution:
    """Prefer a provider-reported cost while retaining reproducible inference."""
    canonical = provider_cost_usd if provider_cost_usd is not None else inferred_cost_usd
    source = (
        "provider"
        if provider_cost_usd is not None
        else ("inferred" if inferred_cost_usd is not None else "unavailable")
    )
    delta = None
    pct = None
    if provider_cost_usd is not None and inferred_cost_usd is not None:
        delta = inferred_cost_usd - provider_cost_usd
        if provider_cost_usd != 0:
            pct = delta / provider_cost_usd
    return CostResolution(
        cost_usd=canonical,
        cost_source=source,
        provider_cost_usd=provider_cost_usd,
        inferred_cost_usd=inferred_cost_usd,
        cost_delta_usd=delta,
        cost_delta_pct=pct,
    )


def rates_payload(rates: Rates | None) -> dict[str, float] | None:
    """The rates as they go into ``result.json`` — the run's frozen price provenance."""
    return None if rates is None else asdict(rates)


def price_provenance(lookup: RateLookup | None) -> dict[str, Any]:
    """The ``result.json`` fields describing what a run was priced by.

    Recorded per run rather than per pass because a matrix can mix models across layers:
    one on negotiated config rates, another on what the page published that morning.
    Without the source and date, two runs months apart are indistinguishable from two
    priced by the same stale table, which is the whole reason for freezing rates at all.
    """
    return {
        "price_rates": rates_payload(None if lookup is None else lookup.rates),
        "price_source": None if lookup is None else lookup.source,
        "price_rates_as_of": None if lookup is None else lookup.as_of,
    }
