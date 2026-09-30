"""Bounded scheduling and hosted-endpoint plumbing for generation stages."""

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from email.utils import parsedate_to_datetime
from random import uniform

import httpx


class UsageLimitError(Exception):
    """The teacher subscription cannot accept more requests in this lane."""


def check_usage_limit(response):
    if response.status_code < 400:
        return
    response.read()
    text = response.text.casefold()
    markers = ("usage limit", "usage_limit", "quota", "insufficient balance",
               "credit balance", "weekly limit", "monthly limit", "daily limit")
    if response.status_code == 402 or any(marker in text for marker in markers):
        # Do not include the body: providers can echo credentials in error responses.
        raise UsageLimitError(f"HTTP {response.status_code}: teacher usage limit reached")


class Throttle:
    """Client-side requests-per-minute limiter shared across worker threads."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            at = max(time.time(), self.next_at)
            self.next_at = at + self.interval
        delay = at - time.time()
        if delay > 0:
            time.sleep(delay)


def api_key_from_env(name):
    """Read a key from an environment variable; it is only ever sent as Bearer."""
    if not name:
        return None
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"environment variable {name} is not set")
    return value


class _Retryable(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


def retry_after(response):
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def backoff_delay(attempt, retry_after_hint=None, cap=60.0):
    delay = min(cap, 2.0**attempt) + uniform(0, 1)
    if retry_after_hint is not None:
        delay = max(delay, min(retry_after_hint, 300.0))
    return delay


def call_with_retries(send, attempts=6):
    """Retry 429, 5xx, timeouts and connection errors with backoff and jitter."""
    for attempt in range(attempts):
        try:
            response = send()
            check_usage_limit(response)
            if response.status_code == 429 or response.status_code >= 500:
                raise _Retryable(response)
            return response
        except httpx.HTTPStatusError:
            raise
        except _Retryable as exc:
            if attempt == attempts - 1:
                if exc.response.status_code == 429:
                    raise UsageLimitError(
                        "HTTP 429: teacher rate limit persisted after retries") from None
                exc.response.raise_for_status()
            delay = backoff_delay(attempt, retry_after(exc.response))
            exc.response.close()
            time.sleep(delay)
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff_delay(attempt))
    raise AssertionError("unreachable")


class TeacherTransport(httpx.BaseTransport):
    """Retry real runtime HTTP calls and expose limits swallowed by the runner."""

    def __init__(self, transport=None, throttle=None, stopped=None):
        self.transport = transport or httpx.HTTPTransport()
        self.throttle = throttle
        self.stopped = stopped if stopped is not None else threading.Event()
        self.error = None

    def handle_request(self, request):
        def send():
            if self.stopped.is_set():
                raise UsageLimitError("teacher lane stopped after usage limit")
            if self.throttle is not None:
                self.throttle.wait()
            response = self.transport.handle_request(request)
            response.request = request
            return response

        try:
            return call_with_retries(send)
        except UsageLimitError as exc:
            self.stopped.set()
            self.error = exc
            raise

    def raise_if_limited(self):
        if self.error is not None:
            raise self.error

    def close(self):
        # Each runtime request closes its Client, but workers share this transport.
        pass

    def __exit__(self, *args):
        self.transport.close()


def post_json(url, payload, headers=None, timeout=180, transport=None):
    def send():
        with httpx.Client(transport=transport, timeout=timeout, trust_env=False) as client:
            return client.post(url, json=payload, headers=headers or {})

    response = call_with_retries(send)
    response.raise_for_status()
    return response.json()


def add_hosted_args(parser):
    """Shared flags for hosted OpenAI-compatible endpoints."""
    parser.add_argument("--api-key-env", default=None,
                        help="environment variable holding the API key")
    parser.add_argument("--json-mode", choices=["schema", "object", "none"], default="schema")
    parser.add_argument("--teacher", default="local", help="label written into output records")
    parser.add_argument("--rpm", type=float, default=None,
                        help="client-side throttle, requests per minute")


def hosted_config(parser, args):
    """Resolve the key (fail fast when the variable is unset) and the throttle."""
    try:
        key = api_key_from_env(args.api_key_env)
    except ValueError as exc:
        parser.error(str(exc))
    return key, Throttle(args.rpm)


def completed(work, items, workers, deadline=None):
    """Yield completed (item, result) pairs; drain in-flight work at the deadline."""
    if workers < 1:
        raise ValueError("workers must be positive")
    items = iter(items)
    skipped = object()
    stopped = threading.Event()
    reported_limit = False

    def start(item):
        if stopped.is_set() or (deadline is not None and time.time() >= deadline):
            return skipped
        try:
            return work(item)
        except UsageLimitError:
            stopped.set()
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                if stopped.is_set() or (deadline is not None and time.time() >= deadline):
                    exhausted = True
                    break
                try:
                    item = next(items)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(start, item)] = item
            if not pending:
                break
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                item = pending.pop(future)
                try:
                    result = future.result()
                except UsageLimitError as exc:
                    if not reported_limit:
                        print(f"LANE_STOP {exc}", flush=True)
                        reported_limit = True
                    exhausted = True
                    continue
                if result is not skipped:
                    yield item, result
