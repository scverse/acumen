"""Render the ``acumen`` CLI reference from its argparse parser.

Two directives, so the reference stays in sync with :func:`acumen.cli.build_parser`:

``.. acumen-commands::``
    An autosummary-style table of the subcommands named in its content (one per line), each
    linking to its page and showing its one-line ``help``.

``.. acumen-command:: NAME``
    A subcommand's page body styled like an autodoc function: the call as the signature, its
    ``description`` and, in the same layout as a function's *Parameters*, its *Options*.
    Directive content, if any, is parsed as the page's own markup and rendered between the
    description and the options.
"""

from __future__ import annotations

import argparse
import re
from functools import cache
from pathlib import Path

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

PROG = "acumen"
_DEFAULT_IN_HELP = re.compile(r"\s*\(default: [^()]*\)\s*$")


@cache
def _subcommands() -> dict[str, tuple[argparse.ArgumentParser, str]]:
    """Return ``{name: (parser, one-line help)}`` for every ``acumen`` subcommand."""
    from acumen.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    helps = {a.dest: a.help or "" for a in sub._choices_actions}
    return {name: (p, helps.get(name, "")) for name, p in sub.choices.items()}


def _lookup(name: str) -> tuple[argparse.ArgumentParser, str]:
    try:
        return _subcommands()[name]
    except KeyError:
        known = ", ".join(sorted(_subcommands()))
        raise ValueError(f"unknown acumen subcommand {name!r} (known: {known})") from None


_LITERAL = re.compile(r"`([^`]+)`")


def _inline(text: str) -> list[nodes.Node]:
    """Turn ``text`` into text nodes, rendering its `backticked` spans as code."""
    out: list[nodes.Node] = []
    pos = 0
    for m in _LITERAL.finditer(text):
        if m.start() > pos:
            out.append(nodes.Text(text[pos : m.start()]))
        out.append(nodes.literal("", m.group(1)))
        pos = m.end()
    if pos < len(text):
        out.append(nodes.Text(text[pos:]))
    return out


def _kind(action: argparse.Action) -> str:
    """Describe an option's value the way autodoc describes a parameter's type."""
    if action.nargs == 0:
        return "flag"
    if action.choices is not None:
        return "{" + ", ".join(str(c) for c in action.choices) + "}"
    if action.type is Path:
        return "path"
    if action.type in (int, float, str):
        return action.type.__name__
    return "str"


def _option_item(action: argparse.Action) -> nodes.list_item:
    """Return one ``--flag (type, default: x) – help`` item, laid out like an autodoc parameter."""
    name = ", ".join(action.option_strings) if action.option_strings else action.dest
    help_text = action.help or ""
    meta: list[nodes.Node] = [nodes.emphasis("", _kind(action))]
    if isinstance(action, argparse._AppendAction):
        meta.append(nodes.emphasis("", "repeatable"))
    if action.required:
        meta.append(nodes.emphasis("", "required"))
    stripped = _DEFAULT_IN_HELP.sub("", help_text)
    if action.default not in (None, False, argparse.SUPPRESS) and action.nargs != 0 and "default:" not in stripped:
        help_text = stripped
        meta.append(nodes.inline("", "default: ", nodes.literal("", str(action.default))))
    para = nodes.paragraph()
    para += nodes.strong("", name)
    para += nodes.Text(" (")
    for i, m in enumerate(meta):
        if i:
            para += nodes.Text(", ")
        para += m
    para += nodes.Text(") – ")
    para += _inline(help_text)
    return nodes.list_item("", para)


class AcumenCommand(SphinxDirective):
    """``.. acumen-command:: NAME``: one subcommand's reference, autodoc-function style."""

    required_arguments = 1
    has_content = True

    def run(self) -> list[nodes.Node]:
        """Build the signature, description and options of one subcommand."""
        name = self.arguments[0]
        parser, help_text = _lookup(name)
        desc = addnodes.desc(domain="std", objtype="describe", desctype="describe", classes=["std", "describe"])
        sig = addnodes.desc_signature("", "", classes=["sig", "sig-object"])
        sig += addnodes.desc_name("", f"{PROG} {name}")
        sig += addnodes.desc_annotation("", " [OPTIONS]")
        desc += sig
        content = addnodes.desc_content()
        desc += content
        content += nodes.paragraph("", "", *_inline(parser.description or help_text))
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, content)
        options = [a for a in parser._actions if not isinstance(a, argparse._HelpAction)]
        if options:
            items = nodes.bullet_list()
            items += [_option_item(a) for a in options]
            field = nodes.field("", nodes.field_name("", "Options"), nodes.field_body("", items))
            content += nodes.field_list("", field)
        return [desc]


class AcumenCommands(SphinxDirective):
    """``.. acumen-commands::``: a summary table of the subcommands listed in the content."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        """Build the summary table."""
        names = [line.strip() for line in self.content if line.strip()]
        table = nodes.table(classes=["autosummary", "longtable"])
        group = nodes.tgroup(cols=2)
        table += group
        group += nodes.colspec(colwidth=30)
        group += nodes.colspec(colwidth=70)
        body = nodes.tbody()
        group += body
        for name in names:
            _, help_text = _lookup(name)
            link = nodes.reference("", "", nodes.literal("", f"{PROG} {name}"), internal=True, refuri=self._uri(name))
            row = nodes.row()
            row += nodes.entry("", nodes.paragraph("", "", link))
            summary = help_text[:1].upper() + help_text[1:]
            row += nodes.entry("", nodes.paragraph("", summary))
            body += row
        return [table]

    def _uri(self, name: str) -> str:
        """Link to the page ``<name>`` next to the current document."""
        builder = self.env.app.builder
        here = self.env.docname
        target = str(Path(here).parent / name) if "/" in here else name
        return builder.get_relative_uri(here, target)


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the directives."""
    app.add_directive("acumen-command", AcumenCommand)
    app.add_directive("acumen-commands", AcumenCommands)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
