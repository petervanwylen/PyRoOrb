"""A PyRoOrb server relaying its visitor's submission to the identification queue.

The web page's "Submit linkage" is POSTed by the SERVER to the queue (``IDQUEUE_URL``),
so the queue sees every visitor of that server arrive from one address and would
charge them all to one allowance. The relaying server therefore names the visitor
(``X-Idqueue-Client``, the address :func:`pyroorb.web.ratelimit.client_of` gave it) and
proves it is a relay the queue trusts (``X-Idqueue-Relay``): the shared secret
``IDQUEUE_RELAY_KEY`` when set, else a secret this process draws at start -- which is
exactly enough for a queue server that posts to itself, and lets nobody else in.
A header pair without the right secret is ignored: the request is charged to its own
address, like any direct caller.
"""
import hmac
import os
import secrets

RELAY_HEADER = "X-Idqueue-Relay"
CLIENT_HEADER = "X-Idqueue-Client"

_PROCESS_SECRET = secrets.token_urlsafe(32)


def secret():
    """The relay secret: ``IDQUEUE_RELAY_KEY``, else this process's own."""
    return (os.environ.get("IDQUEUE_RELAY_KEY") or "").strip() or _PROCESS_SECRET


def headers(client):
    """The headers a relayed POST carries for visitor ``client`` (none without one)."""
    who = str(client or "").strip()[:64]
    if not who:
        return {}
    return {RELAY_HEADER: secret(), CLIENT_HEADER: who}


def relayed_client(request_headers):
    """The visitor a trusted relay names, or None when the request is not one."""
    offered = (request_headers.get(RELAY_HEADER) or "").strip()
    who = (request_headers.get(CLIENT_HEADER) or "").strip()[:64]
    if not offered or not who:
        return None
    if not hmac.compare_digest(offered.encode("utf-8", "replace"),
                               secret().encode("utf-8", "replace")):
        return None
    return who
