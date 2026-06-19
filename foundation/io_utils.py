"""Minimal I/O helpers.

Only :func:`download_file` is needed by the runtime (``message/components.py`` downloads
remote media for local processing), so this module deliberately stays a single small
helper rather than a general-purpose I/O toolkit.
"""

from __future__ import annotations

import inspect
import ssl
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp
import certifi

from .log import logger

__all__ = ["download_file"]


def _safe_url_for_log(url: str) -> str:
    """Return a URL summary that omits query strings and fragments (signed URLs)."""
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        filename = Path(unquote(parsed.path or "")).name
        suffix = f" file={filename!r}" if filename else ""
        return f"{parsed.scheme} URL host={parsed.netloc!r}{suffix} len={len(url)}"
    return f"URL len={len(url)}"


async def _emit_download_progress(progress_callback, payload: dict) -> None:
    if not progress_callback:
        return
    result = progress_callback(payload)
    if inspect.isawaitable(result):
        await result


async def download_file(
    url: str,
    path: str,
    show_progress: bool = False,
    progress_callback=None,
) -> None:
    """从指定 url 下载文件到指定路径 path"""
    try:
        ssl_context = ssl.create_default_context(
            cafile=certifi.where(),
        )  # 使用 certifi 提供的 CA 证书
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(
            trust_env=True,
            connector=connector,
        ) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=1800)) as resp:
                if resp.status != 200:
                    logger.error(
                        "Failed to download file from %s. HTTP status code: %s",
                        _safe_url_for_log(url),
                        resp.status,
                    )
                total_size = int(resp.headers.get("content-length", 0))
                downloaded_size = 0
                start_time = time.time()
                if show_progress:
                    print(f"Downloading: {_safe_url_for_log(url)} | Size: {total_size / 1024:.2f} KB")
                await _emit_download_progress(
                    progress_callback,
                    {
                        "url": url,
                        "downloaded": 0,
                        "total": total_size,
                        "percent": 0,
                        "speed": 0,
                    },
                )
                with open(path, "wb") as f:
                    while True:
                        chunk = await resp.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        elapsed_time = time.time() - start_time if time.time() - start_time > 0 else 1
                        speed = downloaded_size / 1024 / elapsed_time  # KB/s
                        percent = downloaded_size / total_size if total_size > 0 else 0
                        await _emit_download_progress(
                            progress_callback,
                            {
                                "url": url,
                                "downloaded": downloaded_size,
                                "total": total_size,
                                "percent": percent,
                                "speed": speed,
                            },
                        )
                        if show_progress:
                            print(
                                f"\rProgress: {percent:.2%} Speed: {speed:.2f} KB/s",
                                end="",
                            )
                await _emit_download_progress(
                    progress_callback,
                    {
                        "url": url,
                        "downloaded": downloaded_size,
                        "total": total_size,
                        "percent": 1,
                        "speed": 0,
                    },
                )
    except (aiohttp.ClientConnectorSSLError, aiohttp.ClientConnectorCertificateError):
        # 关闭SSL验证（仅在证书验证失败时作为fallback）
        logger.warning(
            f"SSL certificate verification failed for {_safe_url_for_log(url)}. "
            "Falling back to unverified connection (CERT_NONE). "
        )
        logger.warning(
            f"SSL certificate verification failed for {_safe_url_for_log(url)}. "
            "Falling back to unverified connection (CERT_NONE). "
            "This is insecure and exposes the application to man-in-the-middle attacks. "
            "Please investigate certificate issues with the remote server."
        )
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=ssl_context, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                total_size = int(resp.headers.get("content-length", 0))
                downloaded_size = 0
                start_time = time.time()
                if show_progress:
                    print(f"Size: {total_size / 1024:.2f} KB | URL: {_safe_url_for_log(url)}")
                await _emit_download_progress(
                    progress_callback,
                    {
                        "url": url,
                        "downloaded": 0,
                        "total": total_size,
                        "percent": 0,
                        "speed": 0,
                    },
                )
                with open(path, "wb") as f:
                    while True:
                        chunk = await resp.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        elapsed_time = time.time() - start_time if time.time() - start_time > 0 else 1
                        speed = downloaded_size / 1024 / elapsed_time  # KB/s
                        percent = downloaded_size / total_size if total_size > 0 else 0
                        await _emit_download_progress(
                            progress_callback,
                            {
                                "url": url,
                                "downloaded": downloaded_size,
                                "total": total_size,
                                "percent": percent,
                                "speed": speed,
                            },
                        )
                        if show_progress:
                            print(
                                f"\rProgress: {percent:.2%} Speed: {speed:.2f} KB/s",
                                end="",
                            )
                await _emit_download_progress(
                    progress_callback,
                    {
                        "url": url,
                        "downloaded": downloaded_size,
                        "total": total_size,
                        "percent": 1,
                        "speed": 0,
                    },
                )
    if show_progress:
        print()
