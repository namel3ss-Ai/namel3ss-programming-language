"""Development server management with file watching and rebuild support."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BaseException, Callable, Iterable, List, Optional, Sequence

from .codegen import generate_backend, generate_site
from .loader import load_program
from .parser import N3SyntaxError
from .resolver import ModuleResolutionError, resolve_program


def _format_dev_error(exc: BaseException) -> str:
    formatter = getattr(exc, "format", None)
    if callable(formatter):
        try:
            return formatter()
        except Exception:
            pass
    legacy = getattr(exc, "format_message", None)
    if callable(legacy):
        try:
            return legacy()
        except Exception:
            pass
    return str(exc)


@dataclass
class DevServerStatus:
    app_name: str
    last_build_ok: bool
    last_error: Optional[str] = None


class PollingWatcher:
    """Lightweight cross-platform polling watcher."""

    def __init__(self, paths: Sequence[Path], *, interval: float = 0.75) -> None:
        self._paths = [path.resolve() for path in paths]
        self._interval = interval
        self._listener: Optional[Callable[[List[Path]], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._snapshot = {path: path.stat().st_mtime if path.exists() else 0.0 for path in self._paths}

    def watch(self, listener: Callable[[List[Path]], None]) -> None:
        self._listener = listener

    def start(self) -> None:
        if self._listener is None:
            raise RuntimeError("PollingWatcher requires a listener before starting")
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="namel3ss-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            changed: List[Path] = []
            for path in self._paths:
                try:
                    mtime = path.stat().st_mtime
                except FileNotFoundError:
                    mtime = 0.0
                previous = self._snapshot.get(path)
                if previous is None or mtime != previous:
                    self._snapshot[path] = mtime
                    changed.append(path)
            if changed and self._listener is not None:
                try:
                    self._listener(changed)
                except Exception:  # pragma: no cover - watcher should not crash
                    pass
            if self._stop.wait(self._interval):  # pragma: no branch
                break


class DevAppSession:
    """Controls build and dev-server lifecycle for a single app."""

    def __init__(
        self,
        *,
        name: str,
        source: Path,
        backend_out: Path,
        frontend_out: Path,
        host: str,
        port: int,
        frontend_target: str,
        enable_realtime: bool,
        env: Sequence[str],
        watch: bool = True,
    ) -> None:
        self.name = name
        self.source = source
        self.backend_out = backend_out
        self.frontend_out = frontend_out
        self.host = host
        self.port = port
        self.frontend_target = frontend_target
        self.enable_realtime = enable_realtime
        self.env = list(env)
        self.watch = watch
        self._process: Optional[subprocess.Popen[str]] = None
        self._watcher: Optional[PollingWatcher] = None
        self._status = DevServerStatus(app_name=name, last_build_ok=False)
        self._lock = threading.Lock()

    @property
    def status(self) -> DevServerStatus:
        with self._lock:
            return DevServerStatus(
                app_name=self._status.app_name,
                last_build_ok=self._status.last_build_ok,
                last_error=self._status.last_error,
            )

    def start(self) -> None:
        ok = self._build(initial=True)
        if self.watch:
            self._start_watcher()
        self._spawn_uvicorn()
        if not ok:
            self._print_build_failure()

    def stop(self) -> None:
        if self._watcher:
            self._watcher.stop()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                self._process.kill()
        self._process = None

    def rebuild(self) -> None:
        ok = self._build(initial=False)
        if not ok:
            self._print_build_failure()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _start_watcher(self) -> None:
        watcher = PollingWatcher([self.source])
        watcher.watch(lambda _: self.rebuild())
        watcher.start()
        self._watcher = watcher

    def _spawn_uvicorn(self) -> None:
        env = os.environ.copy()
        for assignment in self.env:
            if "=" in assignment:
                key, value = assignment.split("=", 1)
                env[key] = value
        self.backend_out.mkdir(parents=True, exist_ok=True)
        python_exe = sys.executable
        command = [
            python_exe,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--reload",
        ]
        self._process = subprocess.Popen(command, cwd=self.backend_out, env=env)

    def _build(self, *, initial: bool) -> bool:
        if not self.source.exists():
            with self._lock:
                self._status.last_build_ok = False
                self._status.last_error = f"Source file missing: {self.source}"
            return False
        try:
            program = load_program(self.source.parent)
            resolved = resolve_program(program, entry_path=self.source)
            app = resolved.app
            generate_backend(
                app,
                str(self.backend_out),
                embed_insights=False,
                enable_realtime=self.enable_realtime,
            )
            generate_site(
                app,
                str(self.frontend_out),
                enable_realtime=self.enable_realtime,
                target=self.frontend_target,
            )
            with self._lock:
                self._status.last_build_ok = True
                self._status.last_error = None
        except (N3SyntaxError, ModuleResolutionError, Exception) as exc:
            message = _format_dev_error(exc)
            with self._lock:
                self._status.last_build_ok = False
                self._status.last_error = message
            if initial:
                raise
            return False
        return True

    def _print_build_failure(self) -> None:
        status = self.status
        if status.last_error:
            print(f"[namel3ss] Failed to rebuild '{status.app_name}': {status.last_error}", flush=True)


def summarize_sessions(sessions: Iterable[DevAppSession]) -> None:
    for session in sessions:
        status = session.status
        indicator = "✓" if status.last_build_ok else "✗"
        message = f"{indicator} {session.name} -> http://{session.host}:{session.port}"
        if status.last_error:
            message += f" (last error: {status.last_error})"
        print(message)
