import ipaddress
import socket

import httpx
import pytest

from tholos import __version__, fetch


class Peer:
    def __init__(self, address="93.184.216.34"):
        self.address = address

    def get_extra_info(self, name):
        assert name == "server_addr"
        return self.address, 443


def response(content=b"hello", content_type="text/plain", peer="93.184.216.34", **kwargs):
    return httpx.Response(
        200,
        content=content,
        headers={"content-type": content_type},
        extensions={"network_stream": Peer(peer)},
        **kwargs,
    )


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://[::1]/",
        "http://169.254.169.254/",
        "http://10.0.0.1/",
        "file:///etc/passwd",
        "ftp://site.test/",
        "http://[::ffff:127.0.0.1]/",
        "http://[fc00::1]/",
        "http://public.test:8080/",
        "http://user:password@public.test/",
    ],
)
def test_ssrf(url):
    with pytest.raises((ValueError, OSError)):
        fetch.get(url, httpx.MockTransport(lambda _: pytest.fail("Network was touched")))


def test_private_dns_and_mixed_dns(monkeypatch):
    for addresses in [["10.0.0.1"], ["93.184.216.34", "192.168.1.1"]]:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, ips=addresses, **k: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80)) for ip in ips
            ],
        )
        with pytest.raises(ValueError, match="non-public"):
            fetch.get("https://public.test/", httpx.MockTransport(lambda _: pytest.fail("network")))


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.1.1",
        "224.0.0.1",
        "0.0.0.0",
        "::1",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::ffff:93.184.216.34",
        "bad",
    ],
)
def test_public_address(ip):
    assert not fetch.public(ip)


def test_peer_check(public_dns):
    with pytest.raises(ValueError, match="peer"):
        fetch.get("https://site.test/", httpx.MockTransport(lambda _: response(peer="10.0.0.1")))
    with pytest.raises(ValueError, match="peer"):
        fetch.get("https://site.test/", httpx.MockTransport(lambda _: httpx.Response(200)))


def test_redirect_private(public_dns, monkeypatch):
    calls = []

    def resolve(host, *args, **kwargs):
        ip = "10.0.0.1" if host == "internal.test" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)

    def handler(request):
        calls.append((str(request.url), request.headers["host"]))
        return httpx.Response(
            302,
            headers={"location": "https://internal.test/"},
            extensions={"network_stream": Peer()},
        )

    with pytest.raises(ValueError, match="non-public"):
        fetch.get("https://site.test/", httpx.MockTransport(handler))
    assert calls == [("https://93.184.216.34/", "site.test")]


def test_redirect_limit_and_relative(public_dns):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(
            302, headers={"location": "/next"}, extensions={"network_stream": Peer()}
        )

    with pytest.raises(ValueError, match="redirect"):
        fetch.get("https://site.test/", httpx.MockTransport(handler))
    assert len(calls) == 4 and calls[1] == "https://93.184.216.34/next"


def test_content_and_html(public_dns):
    content = (
        b"<title>Research &amp; notes</title><nav>hide</nav><script>hide</script>"
        b"<style>hide</style><p>Hello <a href='/x'>world</a>.</p><footer>hide</footer>"
    )
    result = fetch.get(
        "https://site.test/",
        httpx.MockTransport(lambda _: response(content, "text/html; charset=utf-8")),
    )
    assert result["title"] == "Research & notes"
    assert result["text"] == "Research & notes Hello world ."
    for kind in fetch.CONTENT_TYPES - {"text/html"}:
        assert (
            fetch.get(
                "https://site.test/",
                httpx.MockTransport(lambda _, kind=kind: response(content_type=kind)),
            )["text"]
            == "hello"
        )
    with pytest.raises(ValueError, match="content type"):
        fetch.get(
            "https://site.test/", httpx.MockTransport(lambda _: response(content_type="image/png"))
        )


def test_size_cap(public_dns):
    with pytest.raises(ValueError, match="2 MB"):
        fetch.get(
            "https://site.test/",
            httpx.MockTransport(lambda _: response(b"x" * (fetch.MAX_BYTES + 1))),
        )

    def handler(_):
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"x" * (fetch.MAX_BYTES + 1)),
            headers={"content-type": "text/plain"},
            extensions={"network_stream": Peer()},
        )

    with pytest.raises(ValueError, match="2 MB"):
        fetch.get("https://site.test/", httpx.MockTransport(handler))


def test_fixtures(monkeypatch):
    monkeypatch.setattr(
        fetch, "FIXTURES", {"https://news.test/": "<title>News</title><p>Story</p>"}
    )
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("DNS touched"))
    assert fetch.get("https://news.test/")["title"] == "News"
    with pytest.raises(ValueError, match="not found"):
        fetch.get("https://news.test/missing")
    with pytest.raises(ValueError):
        fetch.get("ftp://news.test/")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 80))],
    )
    with pytest.raises(ValueError, match="non-public"):
        fetch.get("https://real.example/")


def test_dns_rebinding_uses_only_validated_ip(monkeypatch):
    resolutions = []

    def resolve(host, *args, **kwargs):
        resolutions.append(host)
        ip = "93.184.216.34" if len(resolutions) == 1 else "10.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]

    def handler(request):
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "news.test"
        assert request.extensions["sni_hostname"] == "news.test"
        return response()

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    result = fetch.get("https://news.test/story", httpx.MockTransport(handler))
    assert result["url"] == "https://news.test/story"
    assert resolutions == ["news.test"]


def test_pinned_ipv6_and_redirect_identity(monkeypatch):
    address = "2606:4700:4700::1111"
    resolutions = []
    requests = []

    def resolve(host, *args, **kwargs):
        resolutions.append(host)
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, 443, 0, 0))]

    def handler(request):
        requests.append(request)
        assert str(request.url).startswith(f"https://[{address}]/")
        if len(requests) == 1:
            return httpx.Response(
                302,
                headers={"location": "https://other.test/page"},
                extensions={"network_stream": Peer(address)},
            )
        return response(peer=address)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    result = fetch.get("https://news.test/", httpx.MockTransport(handler))
    assert result["url"] == "https://other.test/page"
    assert resolutions == ["news.test", "other.test"]
    assert [request.headers["host"] for request in requests] == resolutions
    assert [request.extensions["sni_hostname"] for request in requests] == resolutions


@pytest.mark.parametrize("family", ["6to4", "teredo", "nat64"])
@pytest.mark.parametrize("embedded", ["10.0.0.1", "127.0.0.1", "169.254.169.254"])
def test_private_ipv4_embedded_in_ipv6(monkeypatch, family, embedded):
    ipv4 = int(ipaddress.IPv4Address(embedded))
    if family == "6to4":
        address = ipaddress.IPv6Address((0x2002 << 112) | (ipv4 << 80) | 1)
        assert str(address.sixtofour) == embedded
    elif family == "teredo":
        address = ipaddress.IPv6Address((0x20010000 << 96) | (ipv4 ^ 0xFFFFFFFF))
        assert str(address.teredo[1]) == embedded
    else:
        address = ipaddress.IPv6Address(int(fetch.NAT64.network_address) | ipv4)
        assert address in fetch.NAT64
    assert not fetch.public(str(address))
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (str(address), 443, 0, 0))
        ],
    )
    with pytest.raises(ValueError, match="non-public"):
        fetch.get(
            "https://news.test/", httpx.MockTransport(lambda _: pytest.fail("Network touched"))
        )


def test_request_identifies_client_and_accepted_content(public_dns):
    def handler(request):
        assert request.headers["user-agent"] == (
            f"Tholos/{__version__} (+https://github.com/mertkayacs/tholos)"
        )
        assert request.headers["accept"] == "text/html,text/plain,application/json;q=0.9,*/*;q=0.1"
        return response()

    assert fetch.get("https://news.test/", httpx.MockTransport(handler))["text"] == "hello"
