"""Tests for the agent_runtime logging facility.

These tests cover library logging etiquette (NullHandler, zero import-time side
effects) and the optional standalone ``configure_logging`` enhancements.

The "pristine import state" assertions run the import in a **fresh subprocess**:
the shared ``conftest.py`` calls ``configure_logging("WARNING")`` at session
scope, which mutates the process-global ``agent_runtime`` logger. Only a clean
subprocess can faithfully observe what merely importing the package does.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_in_subprocess(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a fresh interpreter, returning the completed process.

    The body should print a single line the test can assert on. Failures surface
    as a non-zero return code with the traceback on stderr.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


class TestNullHandlerEtiquette:
    def test_import_leaves_single_null_handler_and_propagate_true(self) -> None:
        proc = _run_in_subprocess(
            """
            import logging
            import agent_runtime
            from agent_runtime import logger

            handlers = logger.handlers
            null_handlers = [h for h in handlers if isinstance(h, logging.NullHandler)]
            assert len(handlers) == 1, f"expected exactly 1 handler, got {handlers!r}"
            assert len(null_handlers) == 1, f"the sole handler must be a NullHandler, got {handlers!r}"
            assert logger.propagate is True, "import MUST NOT disable propagate"
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip().endswith("OK")

    def test_logging_without_host_config_is_silent(self) -> None:
        # With only a NullHandler and no root config, emitting a record must not
        # print the legacy "No handlers could be found" warning nor hit stderr.
        proc = _run_in_subprocess(
            """
            import agent_runtime
            from agent_runtime import logger

            logger.info("hello from a host that never configured logging")
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        assert proc.stdout.strip() == "OK"
        assert "No handlers could be found" not in proc.stderr
        assert proc.stderr.strip() == ""

    def test_record_propagates_to_host_root_handler(self) -> None:
        # Simulate a host that attaches a handler to the root
        # logger. Because the package keeps propagate=True at import and owns no
        # output handler of its own, agent_runtime records must bubble up.
        proc = _run_in_subprocess(
            """
            import logging
            import agent_runtime
            from agent_runtime import logger

            seen = []

            class CaptureHandler(logging.Handler):
                def emit(self, record):
                    seen.append(record.getMessage())

            root = logging.getLogger()
            root.addHandler(CaptureHandler())
            root.setLevel(logging.DEBUG)

            logger.warning("bubbles up to host root")
            assert "bubbles up to host root" in seen, f"record did not propagate: {seen!r}"
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip().endswith("OK")


class TestConfigureLogging:
    def test_level_from_env_var_when_unspecified(self) -> None:
        # No explicit level + AGENT_RUNTIME_LOG_LEVEL=DEBUG => DEBUG.
        proc = _run_in_subprocess(
            """
            import logging, os
            os.environ["AGENT_RUNTIME_LOG_LEVEL"] = "DEBUG"
            from agent_runtime.foundation.log import configure_logging
            from agent_runtime import logger

            configure_logging()
            assert logger.level == logging.DEBUG, f"got {logging.getLevelName(logger.level)}"
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip().endswith("OK")

    def test_explicit_level_overrides_env_var(self) -> None:
        proc = _run_in_subprocess(
            """
            import logging, os
            os.environ["AGENT_RUNTIME_LOG_LEVEL"] = "DEBUG"
            from agent_runtime.foundation.log import configure_logging
            from agent_runtime import logger

            configure_logging("ERROR")
            assert logger.level == logging.ERROR, f"got {logging.getLevelName(logger.level)}"
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip().endswith("OK")

    def test_optional_rotating_file_handler(self, tmp_path) -> None:
        log_file = tmp_path / "nested" / "app.log"
        proc = _run_in_subprocess(
            f"""
            import logging
            from logging.handlers import RotatingFileHandler
            from agent_runtime.foundation.log import configure_logging
            from agent_runtime import logger

            configure_logging(log_file={str(log_file)!r}, max_bytes=1024, backup_count=2)
            rotating = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
            assert len(rotating) == 1, f"expected one RotatingFileHandler, got {{logger.handlers!r}}"
            h = rotating[0]
            assert h.maxBytes == 1024, h.maxBytes
            assert h.backupCount == 2, h.backupCount
            logger.error("written to file sink")
            logging.shutdown()
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert log_file.exists(), "parent dir + log file must be created"
        assert "written to file sink" in log_file.read_text()

    def test_idempotent_no_duplicate_handlers(self) -> None:
        proc = _run_in_subprocess(
            """
            import logging
            from agent_runtime.foundation.log import configure_logging
            from agent_runtime import logger

            configure_logging("INFO")
            after_first = list(logger.handlers)
            configure_logging("INFO")
            after_second = list(logger.handlers)
            assert len(after_second) == len(after_first), (
                f"second call added handlers: {after_first!r} -> {after_second!r}"
            )
            stream = [h for h in logger.handlers if type(h) is logging.StreamHandler]
            assert len(stream) == 1, f"exactly one console handler expected: {logger.handlers!r}"
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip().endswith("OK")

    def test_console_formatter_includes_source_location(self) -> None:
        # The standalone formatter must carry filename:lineno (parity with
        # host's [file:line]); plain stdlib format, no color.
        proc = _run_in_subprocess(
            """
            import logging
            from agent_runtime.foundation.log import configure_logging
            from agent_runtime import logger

            configure_logging("DEBUG")
            stream = [h for h in logger.handlers if type(h) is logging.StreamHandler]
            assert stream, "console handler missing"
            fmt = stream[0].formatter._fmt
            assert "%(filename)s" in fmt and "%(lineno)d" in fmt, f"no source location in {fmt!r}"
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip().endswith("OK")

    def test_sets_propagate_false_when_called(self) -> None:
        proc = _run_in_subprocess(
            """
            from agent_runtime.foundation.log import configure_logging
            from agent_runtime import logger

            assert logger.propagate is True, "precondition: import keeps propagate True"
            configure_logging("INFO")
            assert logger.propagate is False, "standalone config must disable propagate"
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip().endswith("OK")

    def test_noisy_loggers_quieted_only_inside_configure(self) -> None:
        # Import alone must NOT touch third-party logger levels; only an explicit
        # configure_logging() call may quiet them.
        proc = _run_in_subprocess(
            """
            import logging
            import agent_runtime  # noqa: F401  (import must not quiet anyone)

            before = logging.getLogger("httpx").level
            assert before == logging.NOTSET, f"import must not set httpx level, got {before}"

            from agent_runtime.foundation.log import configure_logging
            configure_logging("DEBUG")
            after = logging.getLogger("httpx").level
            assert after == logging.WARNING, f"configure should quiet httpx to WARNING, got {after}"
            print("OK")
            """
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert proc.stdout.strip().endswith("OK")
