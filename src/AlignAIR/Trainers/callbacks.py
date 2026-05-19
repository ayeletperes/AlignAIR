"""Custom Keras callbacks used by AlignAIR training."""

import json
import logging
import os
import subprocess
import sys
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


# Fernsicht (https://github.com/MuteJester/Fernsicht) is a remote progress
# viewer that streams per-epoch metrics to a WebRTC-hosted room. Importing
# fernsicht in the same process as TensorFlow segfaults model.fit — the
# aiortc transport's asyncio event loop races TF's training threads in
# the C runtime.
#
# We isolate by running fernsicht in a small driver subprocess. Main
# process:
#   - spawns the driver, captures the room URL on its stdout
#   - prints the URL banner to stdout (so it lands in SLURM logs)
#   - writes one JSON line of per-epoch metrics to the driver's stdin
#   - closes the pipe at train end; driver flushes + exits
# Main never imports fernsicht, so its memory has no aiortc threads.


_DRIVER = r"""
import sys, os, json
from fernsicht import manual
total = int(os.environ.get('FERNSICHT_TOTAL', '0')) or None
desc = os.environ.get('FERNSICHT_DESC', 'training')
bar = manual(total=total, desc=desc, unit='ep')
sys.stdout.write(getattr(bar, 'url', '') + '\n')
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        logs = json.loads(line)
    except Exception:
        continue
    postfix = {k: float(v) for k, v in logs.items()
               if isinstance(v, (int, float))}
    try:
        bar.update(1, **postfix)
    except Exception:
        pass
try:
    bar.close()
except Exception:
    pass
"""


class FernsichtCallback(tf.keras.callbacks.Callback):
    """Stream per-epoch metrics to a Fernsicht remote progress room.

    Spawns a subprocess that owns the Fernsicht/WebRTC connection, so
    fernsicht is never imported in the main TF process (avoids the
    asyncio + aiortc / TF training-thread race that segfaults
    ``model.fit``).

    Args:
        desc: Short label shown alongside the remote bar.
        total_epochs: Total epochs the bar should advance through.
        disable: Force-disable even if Fernsicht is installed.
    """

    # Metrics piped to the driver each epoch — keep this list small,
    # fernsicht's viewer truncates long postfix strings.
    _METRICS_TO_FORWARD = (
        "loss", "val_loss", "segmentation_loss",
        "classification_loss", "v_allele_auc", "j_allele_auc",
    )

    def __init__(self, desc: str = "training", total_epochs: Optional[int] = None,
                 disable: bool = False):
        super().__init__()
        self.desc = desc
        self.total_epochs = total_epochs
        self.disable = disable
        self._proc: Optional[subprocess.Popen] = None
        self._url: Optional[str] = None

        if self.disable:
            return

        env = os.environ.copy()
        env["FERNSICHT_DESC"] = desc
        env["FERNSICHT_TOTAL"] = str(total_epochs or 0)

        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", "-c", _DRIVER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Fernsicht driver failed to spawn (%s); skipping URL.", exc)
            self.disable = True
            return

        # First line on the driver's stdout is the room URL. Reading this
        # blocks until the driver has established its session — usually a
        # second or two; cap it to keep training startup snappy if the
        # fernsicht server is unreachable.
        url_line = self._read_url_with_timeout(timeout_sec=10.0)
        if not url_line:
            logger.info("Fernsicht driver produced no URL within 10s; skipping.")
            self._terminate_driver()
            self.disable = True
            return

        self._url = url_line.strip()
        banner = "=" * 70
        print(
            f"\n{banner}\n[fernsicht] monitor this run at: {self._url}\n{banner}\n",
            flush=True,
        )

    def _read_url_with_timeout(self, timeout_sec: float) -> Optional[str]:
        """Block-read the driver's first stdout line with a soft timeout.

        Uses a thread because subprocess pipes don't expose a timeout
        directly. Falls back to None on timeout / EOF.
        """
        import threading
        result: dict = {}

        def _reader():
            try:
                if self._proc and self._proc.stdout:
                    result["line"] = self._proc.stdout.readline()
            except Exception as exc:  # noqa: BLE001
                result["err"] = exc

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout_sec)
        if t.is_alive():
            return None
        return result.get("line")

    def on_epoch_end(self, epoch, logs=None):
        if self._proc is None or self._proc.poll() is not None:
            return
        logs = logs or {}
        postfix = {
            k: float(v) for k, v in logs.items()
            if k in self._METRICS_TO_FORWARD and isinstance(v, (int, float))
        }
        if not postfix:
            return
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(postfix) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, AssertionError) as exc:
            logger.debug("FernsichtCallback pipe write failed (%s); disabling.", exc)
            self._terminate_driver()

    def on_train_end(self, logs=None):
        self._terminate_driver()

    def _terminate_driver(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        self._proc = None
