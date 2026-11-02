"""Playwright adapter for the narrow WhatsApp page surface (Stage 8, Slice 8.4b).

This module contains the *mechanics* only: how to read a WhatsApp Web chat
header, how to stage a file, how to find the nodes that name a staged file and
how to find the Send control.  Every decision about whether those findings are
good enough lives in :mod:`jarvis.browser.whatsapp_web`, which is what makes
the policy unit-testable without a browser.

Nothing here is a generic DOM handle: the adapter exposes exactly the
:class:`~jarvis.browser.whatsapp_web.WhatsAppPage` methods and nothing else.
Playwright is never imported at module import time - the page object is passed
in - so this module can be imported on any platform.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from jarvis.browser.whatsapp_web import (
    MAX_CONVERSATION_NAME_CHARS,
    MAX_CONVERSATION_PREVIEW_CHARS,
    MAX_CONVERSATION_TIMESTAMP_CHARS,
    MAX_RECENT_MESSAGES,
    MAX_UNREAD_CONVERSATIONS,
    AttachmentCandidate,
    AttachmentScan,
    ConversationSummary,
    SearchHit,
    SendControl,
    WhatsAppError,
    WhatsAppMessage,
    WhatsAppNotReady,
    WhatsAppSendUncertain,
    WhatsAppSessionState,
)

__all__ = ["PlaywrightWhatsAppPage"]

logger = logging.getLogger("jarvis.browser.whatsapp.page")

#: The chat panel of the active conversation.
_PANEL_SELECTOR = "#main"

#: Regions that make up the composer (the message box and its attachments).
_COMPOSER_SELECTORS = (
    "footer",
    '[data-testid*="composer"]',
    '[role="dialog"]',
    '[data-testid*="preview"]',
    '[data-testid*="dismissible-input"]',
)

#: The search box in the left-hand pane.
_SEARCH_INPUT_SELECTORS = (
    '[data-testid="search-input"]',
    'input[aria-label="Search or start a new chat"]',
    'div[contenteditable="true"][role="textbox"]',
    'input[placeholder*="search" i]',
)

#: Candidate result rows in the left-hand pane.
_SEARCH_RESULT_SELECTORS = (
    '[role="listitem"]',
    '[data-testid="cell-frame"]',
    '[role="button"]',
)

#: W6: candidate conversation rows in the left-hand pane.  These are the same
#: structural rows the search already treats as results; discovery reads them
#: in place and never clicks one.
_CONVERSATION_ROW_SELECTORS = (
    '[role="listitem"]',
    '[data-testid="cell-frame"]',
    '[role="row"]',
)

#: W6: nodes WhatsApp uses to mark a conversation unread or expose a count.
#: The page reports only what it actually finds; no unread state is inferred
#: from a missing badge.
_UNREAD_BADGE_SELECTORS = (
    '[data-testid="icon-unread-count"]',
    '[data-icon="unread"]',
    '[data-icon*="unread" i]',
    '[aria-label*="unread" i]',
)

#: W6: nodes that may hold the latest visible message preview.
_CONVERSATION_PREVIEW_SELECTORS = (
    '[data-testid="last-msg-status"]',
    '[data-testid="msg-preview"]',
    '[data-testid="last-message"]',
)

#: W6: nodes that may hold the conversation preview timestamp.
_CONVERSATION_TIME_SELECTORS = (
    "time",
    '[data-testid="last-msg-time"]',
    '[data-testid="cell-frame-primary-detail"]',
)

#: Candidate message-container groups, in priority order.  WhatsApp's markup
#: is not stable, so the first group that matches anything is used; the in/out
#: group is combined so a fallback never loses one direction.  Every group is
#: scoped to the open conversation panel (``#main``).
_MESSAGE_CONTAINER_GROUPS = (
    '[data-testid="msg-container"]',
    "div.message-in, div.message-out",
    '#main [role="row"]',
)

#: The node that holds a message's visible text.
_MESSAGE_TEXT_SELECTOR = "span.selectable-text, [data-testid=\"msg-text\"]"

#: Per-message bounds: a read is a summary surface, not a full archive.
_MAX_MESSAGE_TEXT_CHARS = 2000
_MAX_MESSAGE_PRE_CHARS = 240
_MAX_MESSAGE_META_CHARS = 80
#: Per-message bounds on the status hints collected from the meta node.
_MAX_STATUS_HINTS = 8
_MAX_STATUS_HINT_CHARS = 80

#: ``data-pre-plain-text`` is ``"[time, date] sender: "``.
_PRE_PLAIN_TEXT_RE = re.compile(r"^\[(?P<timestamp>[^\]]*)\]\s*(?P<sender>.*?):\s*$", re.DOTALL)

#: The hidden file input WhatsApp uses for attachments (approach B).
_FILE_INPUT_SELECTOR = 'input[type="file"]'

#: The message composer.  Scoped to the open conversation's footer first so the
#: left-hand search box (also a contenteditable textbox) can never be filled.
_COMPOSER_INPUT_SELECTORS = (
    '#main footer [contenteditable="true"][role="textbox"]',
    '#main footer [data-testid="conversation-compose-box-input"]',
    '#main footer [contenteditable="true"]',
    '[data-testid="conversation-compose-box-input"]',
    '[contenteditable="true"][data-tab="10"]',
)

#: Hard bound on the composer text a read may return (the tool's own message cap
#: is far smaller; this only keeps a hostile page from returning a transcript).
_MAX_COMPOSER_TEXT_CHARS = 65536

#: Every element that can act as a control.  Controls are *enumerated* and
#: judged by their accessible name - no selector is guessed for "the Send
#: button", because WhatsApp changes its markup freely (the probe showed the
#: control only as an accessible name, not as a stable selector).
_CONTROL_SELECTOR = 'button, [role="button"], [role="menuitem"], a[href]'

#: Upper bound on the candidates returned by one DOM scan.
_MAX_CANDIDATES = 40

#: Upper bound on the named controls returned by one scan.
_MAX_CONTROLS = 40

#: How long ``wait_for_ready`` polls before it refuses (seconds).
_READY_DEFAULT_TIMEOUT_S = 20.0
#: Hard ceiling, so a caller can never wait indefinitely.
_READY_MAX_TIMEOUT_S = 45.0
#: Delay between polls: small enough to notice a render, large enough not to spin.
_READY_POLL_INTERVAL_MS = 250

#: The signed-in application shell.  ``#pane-side`` is the left-hand chat list
#: and exists whenever a device is linked; an open conversation's header is
#: equally good evidence that the session is signed in.
_AUTHENTICATED_SELECTORS = (
    "#pane-side",
    '[data-testid="chat-list"]',
    "#main header",
)

#: The QR sign-in surface.  Only a QR node counts - not the surrounding copy.
_QR_SURFACE_SELECTORS = (
    '[data-testid="qrcode"]',
    'canvas[aria-label*="qr" i]',
    '[data-testid*="qr-code" i]',
)

#: Explicit logged-out markers.  A page that merely lacks a chat list is never
#: reported as logged out: it stays ``unknown`` (fail closed).
_LOGGED_OUT_SELECTORS = (
    '[data-testid="logged-out"]',
    '[data-testid="auth-failure"]',
    '[data-testid="session-expired"]',
)

#: English copy WhatsApp uses on its logged-out screen.  Read only from a small
#: slice of body text, and only when neither the shell nor a QR node exists, so
#: a chat transcript can never be read for classification.
_LOGGED_OUT_PHRASES = (
    "logged out",
    "use whatsapp on your phone",
    "click to reload",
)
_MAX_LOGIN_TEXT_CHARS = 400


# ── read-only DOM scripts (never click, never navigate) ───────────────────

_JS_SESSION_STATE = """
(cfg) => {
  const has = (selectors) => {
    for (const selector of selectors) {
      try { if (document.querySelector(selector)) return true; } catch (e) { /* ignore */ }
    }
    return false;
  };
  if (has(cfg.authenticated)) return 'authenticated';
  if (has(cfg.qr)) return 'qr_required';
  if (has(cfg.loggedOut)) return 'logged_out';
  // A chat surface exists but no shell marker matched: never read page text
  // (it could be a transcript) and never claim a state.
  if (document.querySelector('#main')) return 'unknown';
  const body = (document.body && document.body.innerText) || '';
  const text = body.slice(0, cfg.maxText).toLowerCase();
  for (const phrase of cfg.phrases) { if (text.includes(phrase)) return 'logged_out'; }
  return 'unknown';
}
"""

_JS_HEADER = """
() => {
  const header = document.querySelector('#main header')
    || document.querySelector('[data-testid="conversation-header"]');
  if (!header) return '';
  const titled = header.querySelector('span[title]');
  if (titled) {
    const title = (titled.getAttribute('title') || '').trim();
    if (title) return title;
  }
  return (header.innerText || '').trim();
}
"""

_JS_COMPOSER_TEXT = """
(cfg) => {
  for (const selector of cfg.selectors) {
    const node = document.querySelector(selector);
    if (!node) continue;
    return (node.innerText || node.textContent || '').slice(0, cfg.maxChars);
  }
  return '';
}
"""

_JS_RECENT_MESSAGES = """
(cfg) => {
  const panel = document.querySelector(cfg.panel);
  if (!panel) return {panelFound: false, messages: []};
  // Use the first container group that matches anything: WhatsApp changes its
  // markup, but one of these has always named a message node.
  let nodes = [];
  for (const sel of cfg.containers) {
    const found = Array.from(panel.querySelectorAll(sel));
    if (found.length) { nodes = found; break; }
  }
  if (!nodes.length) nodes = Array.from(panel.querySelectorAll('[role="row"]'));
  nodes = nodes.filter((node) => !!node.querySelector(cfg.textSelector));
  const total = nodes.length;
  if (!total) return {panelFound: true, messages: []};
  // Only the most recent `limit` containers are examined, so older message
  // bodies are never read into the result.
  const start = Math.max(0, total - cfg.limit);
  const out = [];
  for (let i = start; i < total && out.length < cfg.limit; i++) {
    const node = nodes[i];
    // A bubble can contain quoted-message text before its own text: keep the
    // last non-empty text node so the actual message wins.
    let text = '';
    for (const textNode of node.querySelectorAll(cfg.textSelector)) {
      const candidate = (textNode.innerText || '').trim();
      if (candidate) text = candidate;
    }
    if (!text) continue;  // media-only or empty container
    const preNode = node.querySelector('[data-pre-plain-text]');
    const pre = preNode ? (preNode.getAttribute('data-pre-plain-text') || '') : '';
    const meta = node.querySelector('[data-testid="msg-meta"]');
    const metaText = meta ? (meta.innerText || '').trim() : '';
    // W5 status evidence: report only what the page exposes (icon names
    // and accessible labels of the status node).  No status is inferred
    // here; the bounded vocabulary mapping happens in Python.
    const statusHints = [];
    const hintScope = meta || node;
    for (const el of Array.from(hintScope.querySelectorAll('[data-icon], [aria-label], [title]'))) {
      for (const value of [el.getAttribute('data-icon'), el.getAttribute('aria-label'), el.getAttribute('title')]) {
        const hint = (value || '').trim();
        if (!hint) continue;
        const clipped = hint.slice(0, cfg.maxHintChars);
        if (!statusHints.includes(clipped)) statusHints.push(clipped);
      }
      if (statusHints.length >= cfg.maxHints) break;
    }
    let outgoing = null;
    if (node.classList.contains('message-out') || node.closest('.message-out')) outgoing = true;
    else if (node.classList.contains('message-in') || node.closest('.message-in')) outgoing = false;
    out.push({
      text: text.slice(0, cfg.maxChars),
      pre: pre.slice(0, cfg.maxPreChars),
      meta: metaText.slice(0, cfg.maxMetaChars),
      hints: statusHints,
      outgoing: outgoing,
    });
  }
  return {panelFound: true, messages: out};
}
"""

_JS_SEARCH_RESULTS = """
(selectors) => {
  const pane = document.querySelector('#pane-side') || document.body;
  const seen = new Set();
  const out = [];
  for (const sel of selectors) {
    for (const node of Array.from(pane.querySelectorAll(sel))) {
      const titled = node.querySelector('span[title]');
      const name = ((titled && titled.getAttribute('title'))
        || node.getAttribute('aria-label') || '').trim();
      if (!name) continue;
      const key = name.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      const lines = (node.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
      out.push({name: name.slice(0, 120), detail: lines.slice(1).join(' ').slice(0, 80)});
      if (out.length >= 20) return out;
    }
  }
  return out;
}
"""

_JS_CONVERSATION_LIST = """
(cfg) => {
  const pane = document.querySelector(cfg.pane) || document.body;
  const seen = new Set();
  const out = [];
  for (const sel of cfg.rows) {
    for (const node of Array.from(pane.querySelectorAll(sel))) {
      const titled = node.querySelector('span[title]');
      const name = ((titled && titled.getAttribute('title'))
        || node.getAttribute('aria-label') || '').trim();
      if (!name) continue;
      const key = name.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);

      let unread = false;
      let unreadCount = null;
      for (const bsel of cfg.badges) {
        const badge = node.querySelector(bsel);
        if (!badge) continue;
        unread = true;
        const label = (badge.getAttribute('aria-label') || badge.getAttribute('title')
          || badge.textContent || '').trim();
        const match = label.match(/(\\d+)/);
        if (match) {
          const n = parseInt(match[1], 10);
          if (Number.isFinite(n) && n > 0) unreadCount = n;
        }
        break;
      }
      if (!unread) {
        const text = (node.textContent || '').toLowerCase();
        if (text.includes('unread')) unread = true;
      }
      if (!unread) continue;

      let preview = '';
      for (const psel of cfg.preview) {
        const el = node.querySelector(psel);
        if (!el) continue;
        const t = (el.getAttribute('aria-label') || el.getAttribute('title')
          || el.innerText || el.textContent || '').trim();
        if (t) { preview = t; break; }
      }

      let timestamp = '';
      for (const tsel of cfg.time) {
        const el = node.querySelector(tsel);
        if (!el) continue;
        if (el.tagName === 'TIME' && el.getAttribute('datetime')) {
          timestamp = el.getAttribute('datetime') || '';
        }
        if (!timestamp) {
          const t = (el.getAttribute('aria-label') || el.getAttribute('title')
            || el.innerText || el.textContent || '').trim();
          if (t) timestamp = t;
        }
        if (timestamp) break;
      }

      out.push({
        name: name.slice(0, cfg.maxName),
        unread: unread,
        unreadCount: unreadCount,
        preview: preview.slice(0, cfg.maxPreview),
        timestamp: timestamp.slice(0, cfg.maxTime),
      });
      if (out.length >= cfg.limit) return out;
    }
  }
  return out;
}
"""

_JS_OPEN_RESULT = """
(cfg) => {
  const selectors = cfg.selectors;
  const pane = document.querySelector('#pane-side') || document.body;
  const wanted = (cfg.name || '').trim().toLowerCase();
  const matches = [];
  for (const sel of selectors) {
    for (const node of Array.from(pane.querySelectorAll(sel))) {
      const titled = node.querySelector('span[title]');
      const label = ((titled && titled.getAttribute('title'))
        || node.getAttribute('aria-label') || '').trim();
      if (label.toLowerCase() === wanted && !matches.includes(node)) matches.push(node);
    }
  }
  if (matches.length !== 1) return matches.length;
  matches[0].click();
  return 1;
}
"""

_JS_STAGED_FILES = """
() => {
  const names = [];
  for (const input of Array.from(document.querySelectorAll('input[type="file"]'))) {
    for (const file of Array.from(input.files || [])) {
      if (file && file.name) names.push(file.name);
    }
  }
  return names;
}
"""

_JS_ATTACHMENT_SCAN = """
(cfg) => {
  const needle = (cfg.needle || '').trim().toLowerCase();
  const panel = document.querySelector(cfg.panel);
  if (!panel) {
    return {panelFound: false, composer: [], panel: [], composerBelow: false, panelBelow: false,
            composerTextLen: 0, panelTextLen: 0};
  }
  const composerRoots = [];
  for (const sel of cfg.composer) {
    const root = panel.querySelector(sel);
    if (root) composerRoots.push(root);
  }
  const inComposer = (el) => composerRoots.some(root => root === el || root.contains(el));
  const hits = [];
  for (const el of Array.from(panel.querySelectorAll('*'))) {
    let source = '';
    let text = '';
    const attrs = [
      ['aria-label', el.getAttribute('aria-label')],
      ['title', el.getAttribute('title')],
      ['alt', el.getAttribute('alt')],
    ];
    for (const pair of attrs) {
      const value = pair[1];
      if (value && value.toLowerCase().includes(needle)) {
        source = pair[0];
        text = value;
        break;
      }
    }
    if (!source && el.children.length === 0) {
      const own = (el.textContent || '').trim();
      if (own && own.toLowerCase().includes(needle)) {
        source = 'text';
        text = own;
      }
    }
    if (!source) continue;
    hits.push({el: el, scope: inComposer(el) ? 'composer' : 'panel', source: source,
               text: text.slice(0, 120)});
    if (hits.length >= cfg.limit) break;
  }
  const commonAncestor = (nodes) => {
    if (!nodes.length) return null;
    let ancestor = nodes[0].parentElement;
    while (ancestor && !nodes.every(n => ancestor === n || ancestor.contains(n))) {
      ancestor = ancestor.parentElement;
    }
    return ancestor;
  };
  const below = (ancestor) => !!(ancestor && ancestor !== panel && panel.contains(ancestor));
  const textLen = (ancestor) => (ancestor ? (ancestor.textContent || '').trim().length : 0);
  const composerEls = hits.filter(h => h.scope === 'composer').map(h => h.el);
  const panelEls = hits.filter(h => h.scope === 'panel').map(h => h.el);
  const composerAncestor = commonAncestor(composerEls);
  const panelAncestor = commonAncestor(panelEls);
  const strip = (h) => ({scope: h.scope, source: h.source, text: h.text});
  return {
    panelFound: true,
    composer: hits.filter(h => h.scope === 'composer').map(strip),
    panel: hits.filter(h => h.scope === 'panel').map(strip),
    composerBelow: below(composerAncestor),
    panelBelow: below(panelAncestor),
    composerTextLen: textLen(composerAncestor),
    panelTextLen: textLen(panelAncestor),
  };
}
"""

_JS_SEND_CONTROLS = """
(cfg) => {
  const controlSelector = cfg.controls;
  const nameOf = (node) => {
    const label = (node.getAttribute('aria-label') || node.getAttribute('title')
      || (node.textContent || '').trim() || '').trim();
    return label.slice(0, 80);
  };
  const collect = (root) => {
    const out = [];
    const seen = new Set();
    for (const node of Array.from(root.querySelectorAll(controlSelector))) {
      const name = nameOf(node);
      if (!name) continue;
      const key = name.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({name: name});
      if (out.length >= cfg.limit) break;
    }
    return out;
  };
  const panel = document.querySelector('#main');
  if (panel) {
    // The composer first: Send lives there, and the composer is small, so the
    // scan cannot be truncated by a long message list.
    for (const sel of cfg.composer) {
      const root = panel.querySelector(sel);
      if (!root) continue;
      const found = collect(root);
      if (found.length) return found;
    }
    const inPanel = collect(panel);
    if (inPanel.length) return inPanel;
  }
  return collect(document.body);
}
"""

_JS_CLICK_SEND = """
(cfg) => {
  const wanted = (cfg.name || '').trim().toLowerCase();
  const matches = [];
  for (const node of Array.from(document.querySelectorAll(cfg.controls))) {
    const label = ((node.getAttribute('aria-label') || node.getAttribute('title')
      || (node.textContent || '').trim())).trim();
    if (label.toLowerCase() === wanted && !matches.includes(node)) matches.push(node);
  }
  if (matches.length !== 1) return matches.length;
  matches[0].click();
  return 1;
}
"""


class PlaywrightWhatsAppPage:
    """A :class:`~jarvis.browser.whatsapp_web.WhatsAppPage` over a Playwright page."""

    def __init__(self, page: Any) -> None:
        """Wrap an existing Playwright page (never starts a browser)."""
        self._page = page

    # ── navigation ─────────────────────────────────────────────────────────

    def url(self) -> str:
        """The current page URL."""
        return str(self._page.url)

    def goto(self, url: str) -> None:
        """Navigate to an absolute URL and wait for the DOM."""
        self._page.goto(url, wait_until="domcontentloaded", timeout=45_000)

    def chat_header(self) -> str:
        """The conversation header text ("" when it cannot be read)."""
        try:
            return str(self._page.evaluate(_JS_HEADER) or "")
        except Exception as exc:  # noqa: BLE001 - an unreadable header is not a pass
            logger.debug("reading the WhatsApp chat header failed: %s", exc)
            return ""

    # ── conversation-list reading (W6, read-only) ──────────────────────────

    def conversation_summaries(self, limit: int) -> tuple[ConversationSummary, ...]:
        """Read the visible unread conversation rows without opening a chat.

        Only the left-hand conversation list (``#pane-side``/chat list) is
        inspected.  The script never clicks, navigates, fills a composer or
        touches a Send control, so no read receipt can change.  Every field is
        returned only when the row exposes it; nothing is invented.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise WhatsAppError("the unread limit must be an integer; refusing to read")
        if not 1 <= limit <= MAX_UNREAD_CONVERSATIONS:
            raise WhatsAppError(
                f"the unread limit must be between 1 and {MAX_UNREAD_CONVERSATIONS}; "
                "refusing to read"
            )
        config = {
            "pane": "#pane-side",
            "rows": list(_CONVERSATION_ROW_SELECTORS),
            "badges": list(_UNREAD_BADGE_SELECTORS),
            "preview": list(_CONVERSATION_PREVIEW_SELECTORS),
            "time": list(_CONVERSATION_TIME_SELECTORS),
            "limit": limit,
            "maxName": MAX_CONVERSATION_NAME_CHARS,
            "maxPreview": MAX_CONVERSATION_PREVIEW_CHARS,
            "maxTime": MAX_CONVERSATION_TIMESTAMP_CHARS,
        }
        try:
            raw = self._page.evaluate(_JS_CONVERSATION_LIST, config)
        except Exception as exc:
            logger.debug("reading the WhatsApp conversation list failed: %s", exc)
            raise WhatsAppError("the WhatsApp conversation list could not be read") from exc
        if not isinstance(raw, list):
            raise WhatsAppError(
                "the WhatsApp conversation list could not be located; refusing to read"
            )
        summaries: list[ConversationSummary] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            if item.get("unread") is not True:
                continue
            raw_count = item.get("unreadCount")
            count = (
                int(raw_count)
                if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count > 0
                else None
            )
            summaries.append(
                ConversationSummary(
                    name=name[:MAX_CONVERSATION_NAME_CHARS],
                    unread=True,
                    unread_count=count,
                    preview=str(item.get("preview") or "").strip()[
                        :MAX_CONVERSATION_PREVIEW_CHARS
                    ],
                    timestamp=str(item.get("timestamp") or "").strip()[
                        :MAX_CONVERSATION_TIMESTAMP_CHARS
                    ],
                )
            )
            if len(summaries) >= limit:
                break
        return tuple(summaries)

    # ── conversation reading ───────────────────────────────────────────────

    def recent_messages(self, limit: int) -> tuple[WhatsAppMessage, ...]:
        """Read at most ``limit`` recent visible text messages (read-only).

        Only the open conversation panel (``#main``) is inspected, and only the
        last ``limit`` message containers.  Media-only or empty containers are
        skipped.  Sender and timestamp are left empty when the page does not
        expose them; nothing is invented.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise WhatsAppError("the message limit must be an integer; refusing to read")
        if not 1 <= limit <= MAX_RECENT_MESSAGES:
            raise WhatsAppError(
                f"the message limit must be between 1 and {MAX_RECENT_MESSAGES}; "
                "refusing to read"
            )
        config = {
            "panel": _PANEL_SELECTOR,
            "containers": list(_MESSAGE_CONTAINER_GROUPS),
            "textSelector": _MESSAGE_TEXT_SELECTOR,
            "limit": limit,
            "maxChars": _MAX_MESSAGE_TEXT_CHARS,
            "maxPreChars": _MAX_MESSAGE_PRE_CHARS,
            "maxMetaChars": _MAX_MESSAGE_META_CHARS,
            "maxHints": _MAX_STATUS_HINTS,
            "maxHintChars": _MAX_STATUS_HINT_CHARS,
        }
        try:
            raw = self._page.evaluate(_JS_RECENT_MESSAGES, config)
        except Exception as exc:
            logger.debug("reading the WhatsApp messages failed: %s", exc)
            raise WhatsAppError("the WhatsApp conversation could not be read") from exc
        if not isinstance(raw, dict) or not raw.get("panelFound"):
            raise WhatsAppError(
                "the WhatsApp conversation panel could not be located; refusing to read"
            )
        messages: list[WhatsAppMessage] = []
        for item in raw.get("messages") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            sender, timestamp = _split_pre_plain_text(str(item.get("pre") or ""))
            if not timestamp:
                timestamp = str(item.get("meta") or "").strip()
            outgoing = item.get("outgoing")
            messages.append(
                WhatsAppMessage(
                    text=text[:_MAX_MESSAGE_TEXT_CHARS],
                    sender=sender,
                    timestamp=timestamp[:_MAX_MESSAGE_META_CHARS],
                    is_outgoing=outgoing if isinstance(outgoing, bool) else None,
                    sequence=len(messages),
                    status_hints=tuple(
                        str(hint)[:_MAX_STATUS_HINT_CHARS]
                        for hint in (item.get("hints") or [])[:_MAX_STATUS_HINTS]
                        if isinstance(hint, str)
                    ),
                )
            )
            if len(messages) >= limit:
                break
        return tuple(messages)

    # ── session state ──────────────────────────────────────────────────────

    def session_state(self) -> WhatsAppSessionState:
        """Return the page's authentication state (read-only and bounded)."""
        config = {
            "authenticated": list(_AUTHENTICATED_SELECTORS),
            "qr": list(_QR_SURFACE_SELECTORS),
            "loggedOut": list(_LOGGED_OUT_SELECTORS),
            "phrases": list(_LOGGED_OUT_PHRASES),
            "maxText": _MAX_LOGIN_TEXT_CHARS,
        }
        try:
            raw = self._page.evaluate(_JS_SESSION_STATE, config)
        except Exception as exc:  # noqa: BLE001 - a failed probe is not a pass
            logger.debug("the WhatsApp session probe failed: %s", exc)
            return WhatsAppSessionState.UNKNOWN
        try:
            return WhatsAppSessionState(str(raw or ""))
        except ValueError:
            logger.debug("the WhatsApp session probe returned an unrecognized state")
            return WhatsAppSessionState.UNKNOWN

    def wait_for_ready(
        self, timeout_s: float = _READY_DEFAULT_TIMEOUT_S
    ) -> WhatsAppSessionState:
        """Poll the session state until it is recognizable or the budget ends.

        Returns as soon as the page is ``authenticated``, ``qr_required`` or
        ``logged_out``.  A page that stays ``unknown`` raises
        :class:`~jarvis.browser.whatsapp_web.WhatsAppNotReady`; nothing is
        assumed.  Each poll is preceded by Playwright's own bounded wait, so this
        never busy-spins.
        """
        timeout = max(1.0, min(float(timeout_s), _READY_MAX_TIMEOUT_S))
        deadline = time.monotonic() + timeout
        while True:
            state = self.session_state()
            if state is not WhatsAppSessionState.UNKNOWN:
                return state
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WhatsAppNotReady(
                    WhatsAppSessionState.UNKNOWN,
                    f"the WhatsApp page did not become ready within {timeout:.0f}s; "
                    "refusing to continue",
                )
            self._page.wait_for_timeout(
                max(1, min(_READY_POLL_INTERVAL_MS, int(remaining * 1000)))
            )

    # ── search ─────────────────────────────────────────────────────────────

    def fill_search(self, query: str) -> None:
        """Type a query into the WhatsApp search box."""
        for selector in _SEARCH_INPUT_SELECTORS:
            try:
                locator = self._page.locator(selector)
                if locator.count() == 0:
                    continue
                locator.first.fill(query)
                return
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                logger.debug("WhatsApp search box %r unusable: %s", selector, exc)
                continue
        raise WhatsAppError("the WhatsApp search box could not be located")

    def search_results(self) -> tuple[SearchHit, ...]:
        """The visible search results (bounded, de-duplicated by name)."""
        try:
            raw = self._page.evaluate(_JS_SEARCH_RESULTS, list(_SEARCH_RESULT_SELECTORS))
        except Exception as exc:  # noqa: BLE001 - no results is a refusal, not a crash
            logger.debug("reading WhatsApp search results failed: %s", exc)
            return ()
        hits: list[SearchHit] = []
        for item in raw or []:
            name = str((item or {}).get("name") or "").strip()
            if not name:
                continue
            hits.append(SearchHit(name=name, detail=str((item or {}).get("detail") or "")))
        return tuple(hits)

    def open_search_result(self, hit: SearchHit) -> None:
        """Open exactly one search result; refuse when it is not unique."""
        config = {"selectors": list(_SEARCH_RESULT_SELECTORS), "name": hit.name}
        try:
            count = self._page.evaluate(_JS_OPEN_RESULT, config)
        except Exception as exc:
            raise WhatsAppError(f"the WhatsApp search result could not be opened: {exc}") from exc
        if int(count) != 1:
            raise WhatsAppError("the WhatsApp search result is no longer unique; refusing")

    # ── composer ───────────────────────────────────────────────────────────

    def fill_composer(self, text: str) -> None:
        """Replace the active chat composer's contents with ``text``.

        Only the composer of the open conversation is tried, so the search box
        can never be filled.  The driver reads the composer back afterwards and
        refuses on any mismatch; this method only reports mechanics.
        """
        if not isinstance(text, str) or not text.strip():
            raise WhatsAppError("the WhatsApp composer needs a non-empty message")
        for selector in _COMPOSER_INPUT_SELECTORS:
            try:
                locator = self._page.locator(selector)
                if locator.count() == 0:
                    continue
                locator.first.fill(text)
                return
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                logger.debug("WhatsApp composer %r unusable: %s", selector, exc)
                continue
        raise WhatsAppError("the WhatsApp message composer could not be located")

    def composer_text(self) -> str:
        """The active chat composer's visible text ("" when unreadable)."""
        config = {
            "selectors": list(_COMPOSER_INPUT_SELECTORS),
            "maxChars": _MAX_COMPOSER_TEXT_CHARS,
        }
        try:
            return str(self._page.evaluate(_JS_COMPOSER_TEXT, config) or "")
        except Exception as exc:  # noqa: BLE001 - an unreadable composer is not a pass
            logger.debug("reading the WhatsApp composer failed: %s", exc)
            return ""

    # ── attachments ────────────────────────────────────────────────────────

    def set_input_files(self, path: str) -> None:
        """Stage a file through the hidden ``input[type=file]`` (approach B)."""
        try:
            locator = self._page.locator(_FILE_INPUT_SELECTOR)
            if locator.count() == 0:
                raise WhatsAppError("no file input was found in the WhatsApp page")
            locator.first.set_input_files(path)
        except WhatsAppError:
            raise
        except Exception as exc:
            raise WhatsAppError(f"the attachment could not be staged: {exc}") from exc

    def staged_file_names(self) -> tuple[str, ...]:
        """The file names the file input currently holds (may be empty)."""
        try:
            names = self._page.evaluate(_JS_STAGED_FILES)
        except Exception as exc:  # noqa: BLE001 - no evidence is not a pass
            logger.debug("reading the staged file names failed: %s", exc)
            return ()
        return tuple(str(name) for name in (names or []) if str(name))

    def attachment_scan(self, filename: str) -> AttachmentScan:
        """Scan the chat panel for the nodes that name ``filename``."""
        config = {
            "panel": _PANEL_SELECTOR,
            "composer": list(_COMPOSER_SELECTORS),
            "needle": filename,
            "limit": _MAX_CANDIDATES,
        }
        try:
            raw = self._page.evaluate(_JS_ATTACHMENT_SCAN, config) or {}
        except Exception as exc:  # noqa: BLE001 - a failed scan is a refusal
            logger.debug("scanning the WhatsApp chat panel failed: %s", exc)
            return AttachmentScan(panel_found=False)
        return AttachmentScan(
            panel_found=bool(raw.get("panelFound")),
            composer=_candidates(raw.get("composer")),
            panel=_candidates(raw.get("panel")),
            composer_container_below_panel=bool(raw.get("composerBelow")),
            panel_container_below_panel=bool(raw.get("panelBelow")),
            composer_container_text_len=int(raw.get("composerTextLen") or 0),
            panel_container_text_len=int(raw.get("panelTextLen") or 0),
        )

    # ── sending ────────────────────────────────────────────────────────────

    def send_controls(self) -> tuple[SendControl, ...]:
        """The named controls of the active chat (bounded).

        Controls are *enumerated* and identified by their accessible name, so
        no selector has to be guessed for WhatsApp's Send button.  The composer
        is scanned first (the Send control lives there and the composer is
        small enough that the scan cannot be truncated).
        """
        config = {
            "controls": _CONTROL_SELECTOR,
            "composer": list(_COMPOSER_SELECTORS),
            "limit": _MAX_CONTROLS,
        }
        try:
            raw = self._page.evaluate(_JS_SEND_CONTROLS, config)
        except Exception as exc:  # noqa: BLE001 - no controls is a refusal
            logger.debug("reading the WhatsApp send controls failed: %s", exc)
            return ()
        controls: list[SendControl] = []
        for index, item in enumerate(raw or []):
            name = str((item or {}).get("name") or "").strip()
            if not name:
                continue
            controls.append(SendControl(name=name, index=index))
        return tuple(controls)

    def click_send(self, name: str) -> None:
        """Click the single control whose accessible name is exactly ``name``.

        An evaluation failure is reported as :class:`WhatsAppSendUncertain`: the
        click may already have reached WhatsApp, so the caller must not treat it
        as "nothing was sent".  A non-unique control is a plain refusal because
        nothing was clicked.
        """
        config = {"controls": _CONTROL_SELECTOR, "name": name}
        try:
            count = self._page.evaluate(_JS_CLICK_SEND, config)
        except Exception as exc:
            raise WhatsAppSendUncertain(
                "the WhatsApp Send control could not be clicked; its outcome is unknown"
            ) from exc
        if int(count) != 1:
            raise WhatsAppError("the WhatsApp Send control is no longer unique; refusing to send")


def _split_pre_plain_text(raw: str) -> tuple[str, str]:
    """Split WhatsApp's ``"[time, date] sender: "`` prefix.

    Returns ``("", "")`` when the prefix is absent or malformed: an unreadable
    sender or timestamp is unavailable, never guessed.
    """
    match = _PRE_PLAIN_TEXT_RE.match(str(raw or "").strip())
    if match is None:
        return "", ""
    return match.group("sender").strip(), match.group("timestamp").strip()


def _candidates(raw: Any) -> tuple[AttachmentCandidate, ...]:
    """Convert the raw scan payload into typed candidates."""
    out: list[AttachmentCandidate] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        out.append(
            AttachmentCandidate(
                scope=str(item.get("scope") or ""),
                source=str(item.get("source") or ""),
                text=str(item.get("text") or "")[:120],
            )
        )
    return tuple(out)
