"""Permanently denied browser hosts (generic browser only).

The generic managed browser (the ``browser_*`` tools, the task scope grant, the
browsing grant and the static ``browser.allowed_domains`` boundary) must never
reach WhatsApp.  WhatsApp is reachable only through the dedicated integration:
the isolated ``whatsapp`` browser profile, the narrow
:class:`~jarvis.browser.whatsapp_web.WhatsAppDriver` surface and its
exact-origin pin.  Denying the host here means a scope grant, a browsing grant,
a dynamic allowed-domain entry or a redirect can never re-open the generic
route around those invariants (docs/03 I10, ROADMAP W1).

Pure and stdlib-only: no settings, no network, no filesystem, so the policy
engine, both grant registries and the browser driver import it without a cycle.

Matching is on the canonical *host label*, never on a substring:
``notwhatsapp.com`` and ``whatsapp.com.attacker.invalid`` are different hosts
and stay allowed.  A value whose host cannot be canonicalised is denied (fail
closed), because an unreadable target must never be treated as "not WhatsApp".
"""

from __future__ import annotations

from urllib.parse import urlsplit

__all__ = [
    "DENIED_BROWSER_HOSTS",
    "canonical_host",
    "denied_host_reason",
    "host_is_denied",
]

#: Hosts denied to the generic browser: the host itself and every true
#: subdomain.  The dedicated WhatsApp profile never consults this list.
DENIED_BROWSER_HOSTS = ("whatsapp.com",)


def canonical_host(value: str) -> str | None:
    """Return the canonical host label of a bare host or URL, else ``None``.

    Lower-cases, strips a trailing dot, rejects userinfo and any value that
    carries a path/query/fragment, so two spellings of one host compare equal.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if "://" in text:
        try:
            parts = urlsplit(text)
        except ValueError:
            return None
        if "@" in parts.netloc:
            return None
        raw = parts.hostname
    else:
        raw = text
    if not isinstance(raw, str):
        return None
    host = raw.strip().lower().rstrip(".")
    if not host or any(ch.isspace() for ch in host) or any(ch in host for ch in "/\\@?#"):
        return None
    return host


def host_is_denied(value: str) -> bool:
    """Whether ``value`` names a denied host or a true subdomain of one.

    An empty, malformed or unparseable value is denied (fail closed).
    """
    host = canonical_host(value)
    if host is None:
        return True
    return any(host == denied or host.endswith("." + denied) for denied in DENIED_BROWSER_HOSTS)


def denied_host_reason(value: str) -> str:
    """The refusal text for a denied host (empty when the host is allowed)."""
    if not host_is_denied(value):
        return ""
    host = canonical_host(value) or "(unparseable)"
    return (
        f"{host} is reserved for the dedicated WhatsApp integration and cannot be "
        "used by the general browser"
    )
