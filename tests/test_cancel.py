#!/usr/bin/env python3
"""
Test whether the server actually stops inference when a client disconnects.

Usage:
    python tests/test_cancel.py [--url URL] [--voice VOICE]

What it does:
  1. Send a streaming TTS request for a long text
  2. Read only 2 chunks, then close the connection abruptly
  3. Immediately send a second request
  4. Measure the TTFA of the second request — if cancel works, TTFA should
     be roughly normal (~300ms); if not, the second request will queue
     behind the first and TTFA will be several seconds.
"""
import argparse
import socket
import time


API_URL = "http://localhost:23456/v1/audio/speech"
LONG_TEXT = (
    "这是一段用于测试中断功能的长文本。"
    "我们需要确保当客户端断开连接时，服务器能够及时停止推理，"
    "而不是继续把整个推理过程跑完。"
    "如果中断机制正常工作，第二个请求的首包延迟应该很短。"
    "反之，如果第一个请求仍然占着 GPU 跑完全部内容，"
    "第二个请求就会排队等待，导致首包延迟非常高。"
    "这段文字足够长，可以让模型生成好几秒的音频。"
)


def test_cancel_via_raw_socket(url: str, voice: str):
    """Use raw sockets to ensure immediate TCP RST on close."""
    import json
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 80
    path = parsed.path

    payload = json.dumps({
        "model": "qwen3-tts",
        "input": LONG_TEXT,
        "voice": voice,
        "stream": True,
        "response_format": "pcm",
    })

    # --- Request 1: send, read 2 chunks, then RST ---
    print(f"[1] Sending streaming request to {host}:{port}{path} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    # Immediately set SO_LINGER to force RST on close, no TIME_WAIT
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b'\x01\x00\x00\x00\x00\x00\x00\x00')

    req = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
        f"{payload}"
    )
    sock.sendall(req.encode())

    # Read just enough to get 2 chunks (read a small buffer)
    chunks_received = 0
    buf = b""
    header_done = False
    t_start = time.monotonic()
    while chunks_received < 2:
        data = sock.recv(4096)
        if not data:
            break
        buf += data
        if not header_done:
            if b"\r\n\r\n" in buf:
                header_done = True
                # For chunked transfer encoding, count chunk separators
        # Count transfer-encoding chunks (each starts with hex size)
        if header_done:
            # Rough heuristic: each PCM chunk is ~15KB, just count data received
            # after header
            header_end = buf.index(b"\r\n\r\n") + 4
            body = buf[header_end:]
            # Each audio chunk is chunk_size=12 frames * 640 samples/frame * 2 bytes = ~15360 bytes
            chunks_received = len(body) // 8000  # rough estimate
    
    elapsed_to_2_chunks = time.monotonic() - t_start
    print(f"[1] Got ~{chunks_received} chunks in {elapsed_to_2_chunks:.3f}s, closing connection (RST)...")
    sock.close()  # With SO_LINGER(0), this sends TCP RST
    t_close = time.monotonic()

    # --- Request 2: measure TTFA ---
    print(f"[2] Immediately sending second request...")
    import httpx
    t2_start = time.monotonic()
    wait_before_req2 = t2_start - t_close
    print(f"    (waited {wait_before_req2*1000:.0f}ms between close and second request)")

    with httpx.Client(timeout=60) as client:
        with client.stream(
            "POST", url,
            json={
                "model": "qwen3-tts",
                "input": "这是第二个请求，测试首包延迟。",
                "voice": voice,
                "stream": True,
                "response_format": "pcm",
            },
        ) as resp:
            first_byte = None
            total_bytes = 0
            for chunk in resp.iter_bytes():
                if first_byte is None:
                    first_byte = time.monotonic()
                    ttfa = first_byte - t2_start
                    print(f"[2] TTFA = {ttfa:.3f}s")
                total_bytes += len(chunk)
            t2_end = time.monotonic()

    print(f"[2] Total: {total_bytes} bytes in {t2_end - t2_start:.3f}s")
    print()

    # Verdict
    if first_byte is None:
        print("FAIL: No data received from second request")
        return

    if ttfa > 2.0:
        print(f"FAIL: TTFA = {ttfa:.3f}s > 2s — cancel likely did NOT work")
        print("      The first request probably ran to completion, blocking the second.")
    elif ttfa > 1.0:
        print(f"WARN: TTFA = {ttfa:.3f}s — cancel may be slow or partially working")
    else:
        print(f"OK:   TTFA = {ttfa:.3f}s — cancel appears to work correctly")


def test_cancel_via_httpx(url: str, voice: str):
    """Use httpx to test cancel — more realistic but may not RST immediately."""
    import httpx

    print(f"[1] Sending streaming request (httpx) ...")
    t1_start = time.monotonic()
    chunks_read = 0
    with httpx.Client(timeout=60) as client:
        with client.stream(
            "POST", url,
            json={
                "model": "qwen3-tts",
                "input": LONG_TEXT,
                "voice": voice,
                "stream": True,
                "response_format": "pcm",
            },
        ) as resp:
            for chunk in resp.iter_bytes():
                chunks_read += 1
                if chunks_read >= 3:
                    elapsed = time.monotonic() - t1_start
                    print(f"[1] Got {chunks_read} chunks in {elapsed:.3f}s, closing...")
                    break
    
    t_close = time.monotonic()
    print(f"[1] Connection closed at {t_close - t1_start:.3f}s")

    # --- Request 2 ---
    print(f"[2] Sending second request...")
    t2_start = time.monotonic()
    with httpx.Client(timeout=60) as client:
        with client.stream(
            "POST", url,
            json={
                "model": "qwen3-tts",
                "input": "这是第二个请求，测试首包延迟。",
                "voice": voice,
                "stream": True,
                "response_format": "pcm",
            },
        ) as resp:
            first_byte = None
            total_bytes = 0
            for chunk in resp.iter_bytes():
                if first_byte is None:
                    first_byte = time.monotonic()
                    ttfa = first_byte - t2_start
                    print(f"[2] TTFA = {ttfa:.3f}s")
                total_bytes += len(chunk)
            t2_end = time.monotonic()

    print(f"[2] Total: {total_bytes} bytes in {t2_end - t2_start:.3f}s")
    print()
    if first_byte and ttfa > 2.0:
        print(f"FAIL: TTFA = {ttfa:.3f}s > 2s — cancel likely did NOT work")
    elif first_byte and ttfa > 1.0:
        print(f"WARN: TTFA = {ttfa:.3f}s — cancel may be slow")
    elif first_byte:
        print(f"OK:   TTFA = {ttfa:.3f}s — cancel appears to work")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=API_URL)
    parser.add_argument("--voice", default="english")
    parser.add_argument(
        "--method", choices=["socket", "httpx", "both"], default="both",
        help="Which test method to use",
    )
    args = parser.parse_args()

    if args.method in ("socket", "both"):
        print("=" * 60)
        print("Test 1: Raw socket with TCP RST")
        print("=" * 60)
        test_cancel_via_raw_socket(args.url, args.voice)
        print()
        # Give server a moment to notice
        time.sleep(1)

    if args.method in ("httpx", "both"):
        print("=" * 60)
        print("Test 2: httpx graceful close")
        print("=" * 60)
        test_cancel_via_httpx(args.url, args.voice)
