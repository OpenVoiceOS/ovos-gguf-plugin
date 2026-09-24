"""Shared test helpers.

The end-to-end tests download tiny GGUF models from the HuggingFace hub.  When
CI runs the build-tests matrix, several Python-version jobs start at the same
time and each pulls the same model files, so HuggingFace occasionally answers
one of them with ``HTTP 429 Too Many Requests``.  That is a transient,
non-deterministic failure -- a different job hits it on each run.

``hf_retry`` wraps the model-acquisition call and retries it with exponential
backoff (plus jitter) when, and only when, the hub returns a rate-limit (429)
response.  On success it is a transparent pass-through, so it never changes
behaviour when the hub is reachable.  Any non-429 error is re-raised
immediately so real bugs still surface.
"""
import random
import time

from ovos_utils.log import LOG

# Backoff schedule in seconds between attempts (so up to len(_BACKOFFS)+1 tries).
_BACKOFFS = (5, 15, 45, 90)


def _status_code_of(exc):
    """Best-effort extraction of an HTTP status code from a hub/HTTP error."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code is not None:
            return code
    # Some libraries expose it directly.
    return getattr(exc, "status_code", None)


def _is_rate_limit(exc):
    """True only for transient HuggingFace rate-limit (429) errors."""
    if _status_code_of(exc) == 429:
        return True
    # Fall back to message sniffing for wrappers that drop the response object.
    text = str(exc)
    return "429" in text and "Too Many Requests" in text


def hf_retry(factory, *, retries=len(_BACKOFFS)):
    """Call ``factory()`` and retry on transient HuggingFace 429 responses.

    ``factory`` is a zero-arg callable that downloads/loads the model and
    returns it.  On a 429 we sleep with exponential backoff and jitter, then
    retry.  Any other exception propagates immediately; the last 429 is
    re-raised once retries are exhausted.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is a 429
            if not _is_rate_limit(exc):
                raise
            last_exc = exc
            if attempt == retries:
                break
            base = _BACKOFFS[attempt]
            delay = base + random.uniform(0, base * 0.5)
            LOG.warning(
                f"HuggingFace returned 429 (attempt {attempt + 1}/{retries + 1}); "
                f"backing off {delay:.1f}s before retry"
            )
            time.sleep(delay)
    raise last_exc
