"""Named web sites the owner lets JARVIS open (Stage 8.6, slice 8.6.1).

Why this exists
---------------
The planner is an LLM: asked to "open YouTube" it will happily *invent* a URL
("https://youtube.com/watch?v=...") and hand it to ``browser_goto``.  That URL
was never checked by anyone, and the domain-scope confirmation then reads like
noise.  A **site directory** moves that decision out of the model: the owner
writes the URLs in ``config.toml`` and JARVIS resolves a *name* to the URL it
was given.

Guarantees
----------
* Only owner-configured entries exist.  A name the owner did not write down
  does not resolve - there is no fallback, no "search for it", no fuzzy guess.
* A configured URL that is not absolute http(s) without userinfo is **dropped**
  at load time (with a warning), so a typo in ``config.toml`` can never become
  a ``file://``/``javascript:`` navigation.
* The directory grants **no** permission.  ``browser_open_site`` produces the
  same browser target as ``browser_goto`` and is decided by the same
  PolicyEngine with the same domain-scope rules; a site is not a blanket
  domain allow-list entry (docs/03 B4/D18).
* The directory is read-only configuration: nothing here writes to it, and it
  never enters the LangGraph checkpoint.
"""

from __future__ import annotations

import logging
import urllib.parse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.policy.rules import url_origin
from jarvis.tools.web import is_allowed_http_url

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SITES",
    "Site",
    "SiteDirectory",
    "render_site_catalogue",
    "sites_from_settings",
]

#: Hard cap on configured sites (a bounded directory keeps the prompt small).
MAX_SITES = 50
#: Hard cap on a site name / alias length.
MAX_KEY_CHARS = 64
#: Hard cap on the number of aliases per site.
MAX_ALIASES = 8


@dataclass(frozen=True)
class Site:
    """One named site: a display name, its canonical URL and optional aliases."""

    name: str
    url: str
    aliases: tuple[str, ...] = ()

    @property
    def host(self) -> str:
        """The URL's host (lower-cased), used in the prompt and the target."""
        return (urllib.parse.urlsplit(self.url).hostname or "").lower()


def _normalise_key(value: str) -> str:
    """Normalise a lookup key: case-folded, whitespace-collapsed, trimmed."""
    return " ".join(str(value or "").split()).casefold()


def _clean_key(value: str) -> str:
    """Normalise a *configured* key and bound its length."""
    return _normalise_key(value)[:MAX_KEY_CHARS]


class SiteDirectory:
    """An immutable name -> :class:`Site` lookup.

    Lookups are exact after normalisation (case and whitespace insensitive).
    The first entry wins when two entries claim the same key, so the order in
    ``config.toml`` is the tie-break - it is deterministic and documented.
    """

    def __init__(self, sites: Iterable[Site] = ()) -> None:
        self._sites: tuple[Site, ...] = tuple(sites)
        by_key: dict[str, Site] = {}
        for site in self._sites:
            for key in (site.name, *site.aliases):
                norm = _clean_key(key)
                if not norm:
                    continue
                by_key.setdefault(norm, site)
        self._by_key = by_key

    def __len__(self) -> int:
        return len(self._sites)

    @property
    def sites(self) -> tuple[Site, ...]:
        """The configured sites, in configuration order."""
        return self._sites

    def resolve(self, name: str) -> Site | None:
        """Return the site for ``name`` (or one of its aliases), else ``None``."""
        return self._by_key.get(_clean_key(name))

    def names(self) -> list[str]:
        """The configured site names, in configuration order."""
        return [site.name for site in self._sites]


def _site_from_entry(entry: object) -> Site | None:
    """Build a :class:`Site` from a config entry, or ``None`` when unusable."""
    name = getattr(entry, "name", "")
    url = getattr(entry, "url", "")
    aliases = getattr(entry, "aliases", ()) or ()
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(url, str) or not url.strip():
        return None
    clean_name = " ".join(name.split())[:MAX_KEY_CHARS]
    clean_url = url.strip()
    if not is_allowed_http_url(clean_url):
        logger.warning(
            "site directory: dropping %r - the URL is not absolute http(s) without userinfo",
            clean_name,
        )
        return None
    if url_origin(clean_url) is None:
        logger.warning("site directory: dropping %r - the URL has no usable origin", clean_name)
        return None
    clean_aliases: list[str] = []
    for alias in aliases:
        if not isinstance(alias, str):
            continue
        key = _clean_key(alias)
        if key and key not in clean_aliases:
            clean_aliases.append(key)
    return Site(name=clean_name, url=clean_url, aliases=tuple(clean_aliases[:MAX_ALIASES]))


def sites_from_settings(settings: Settings) -> SiteDirectory:
    """Build the directory from the ``[sites]`` section of the settings.

    Unusable entries are dropped (with a warning) rather than failing the whole
    directory: one bad line in ``config.toml`` must not disable every other
    site.
    """
    entries: Sequence[object] = getattr(settings, "sites", None) or ()
    sites: list[Site] = []
    for entry in entries:
        if len(sites) >= MAX_SITES:
            logger.warning("site directory: only the first %d sites are used", MAX_SITES)
            break
        site = _site_from_entry(entry)
        if site is not None:
            sites.append(site)
    return SiteDirectory(sites)


def render_site_catalogue(directory: SiteDirectory) -> str:
    """Render the directory for the brain prompt (names and hosts only).

    Only the *name* and the *host* are shown: the model picks a name and the
    code supplies the URL, so it never has to copy a URL out of the prompt.
    """
    if not len(directory):
        return ""
    lines = ["Web sites you can open with browser_open_site (use the name exactly):"]
    for site in directory.sites:
        aliases = ", ".join(site.aliases)
        suffix = f" (also: {aliases})" if aliases else ""
        lines.append(f"- {site.name} [{site.host}]{suffix}")
    return "\n".join(lines)
