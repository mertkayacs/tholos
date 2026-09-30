import ipaddress
import socket
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from tholos import __version__

FIXTURES: dict[str, str] | None = None
MAX_BYTES = 2 * 1024 * 1024
CONTENT_TYPES = {"text/html", "text/plain", "application/json", "application/xml", "text/xml"}
NAT64 = ipaddress.IPv6Network("64:ff9b::/96")


def public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if isinstance(ip, ipaddress.IPv6Address):
            embedded = ip.sixtofour
            if ip.teredo:
                embedded = ip.teredo[1]
            elif ip in NAT64:
                embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
            if embedded is not None and not public(str(embedded)):
                return False
        return (
            ip.is_global
            and not ip.is_multicast
            and not ip.is_reserved
            and not ip.is_unspecified
            and not getattr(ip, "ipv4_mapped", None)
        )
    except ValueError:
        return False


def _validate(url: str, fixture: bool = False) -> str | None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Only http and https URLs are allowed")
    if parts.username or parts.password or (parts.port and parts.port not in {80, 443}):
        raise ValueError("Credentials and ports other than 80 and 443 are refused")
    if fixture:
        return None
    addresses = socket.getaddrinfo(
        parts.hostname,
        parts.port or (443 if parts.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )
    if not addresses or any(not public(info[4][0]) for info in addresses):
        raise ValueError("The URL resolves to a non-public address")
    return addresses[0][4][0]


class _HTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden: list[str] = []
        self.in_title = False
        self.title: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer"}:
            self.hidden.append(tag)
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.hidden:
            return
        if self.in_title:
            self.title.append(data)
        self.text.append(data)


def _result(url: str, text: str, content_type: str) -> dict:
    title = ""
    if content_type == "text/html":
        parser = _HTML()
        parser.feed(text)
        title = " ".join(" ".join(parser.title).split())
        text = " ".join(parser.text)
    return {"url": url, "title": title, "text": " ".join(text.split())[:2400]}


def get(url: str, transport: httpx.BaseTransport | None = None) -> dict:
    deadline = time.monotonic() + 15
    for redirect in range(4):
        fixture = FIXTURES is not None and (urlsplit(url).hostname or "").endswith(".test")
        address = _validate(url, fixture)
        if fixture:
            if url not in FIXTURES:
                raise ValueError("Fixture URL was not found")
            if len(FIXTURES[url].encode("utf-8")) > MAX_BYTES:
                raise ValueError("Response exceeds 2 MB")
            return _result(url, FIXTURES[url], "text/html")
        if time.monotonic() >= deadline:
            raise ValueError("Fetch exceeded 15 seconds")
        original = httpx.URL(url)
        pinned = original.copy_with(host=address)
        # A fresh client keeps each redirect's TLS identity separate, even on the same IP.
        with (
            httpx.Client(
                transport=transport, timeout=15, follow_redirects=False, trust_env=False
            ) as client,
            client.stream(
                "GET",
                pinned,
                headers={
                    "Host": original.netloc.decode("ascii"),
                    "User-Agent": f"Tholos/{__version__} (+https://github.com/mertkayacs/tholos)",
                    "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1",
                },
                extensions={"sni_hostname": original.raw_host.decode("ascii")},
                timeout=max(0.01, deadline - time.monotonic()),
            ) as response,
        ):
            stream = response.extensions.get("network_stream")
            peer = stream.get_extra_info("server_addr") if stream else None
            if not peer or not public(peer[0]):
                raise ValueError("Connected peer is not a public address")
            if response.status_code in {301, 302, 303, 307, 308}:
                if redirect == 3 or "location" not in response.headers:
                    raise ValueError("Too many redirects or missing redirect location")
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type not in CONTENT_TYPES:
                raise ValueError("Response content type is refused")
            if int(response.headers.get("content-length", "0")) > MAX_BYTES:
                raise ValueError("Response exceeds 2 MB")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    raise ValueError("Response exceeds 2 MB")
                if time.monotonic() >= deadline:
                    raise ValueError("Fetch exceeded 15 seconds")
            return _result(url, body.decode(response.encoding or "utf-8", "replace"), content_type)
    raise ValueError("Fetch did not produce a response")
