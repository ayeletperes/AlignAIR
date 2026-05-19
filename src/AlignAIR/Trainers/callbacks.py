"""Custom Keras callbacks used by AlignAIR training."""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


# Fernsicht is an optional remote progress viewer
# (https://github.com/MuteJester/Fernsicht). It runs an asyncio + WebRTC
# background transport which, on some configurations, races TF's training
# threads and segfaults the process partway through model.fit. To stay
# safe by default we:
#   1. Lazy-import fernsicht only when the callback is constructed (not at
#      module load), so plain `import AlignAIR.API.TrainModel` is unaffected.
#   2. Print the room URL during __init__ (which runs in main(), BEFORE
#      model.fit is called) — that way the URL always reaches stdout/SLURM
#      logs even if training crashes later.
#   3. Close the WebRTC transport right after grabbing the URL, so it can't
#      race TF. Cost: no live per-epoch updates pushed to the room. Benefit:
#      stable training. The room URL still resolves to the Fernsicht viewer.


class FernsichtCallback(tf.keras.callbacks.Callback):
    """Generate a Fernsicht room URL and print it to stdout.

    Note: this used to push per-epoch metrics to the room, but the WebRTC
    transport conflicts with TF's training threads (segfault). For now the
    callback only generates the URL and prints it; the room stays valid as
    a viewer landing page but doesn't receive live updates. Watch the
    SLURM stdout (or `tail -f slurm-alignair-aa-*.out`) for the per-epoch
    Keras progress instead.

    Args:
        desc: Short label shown alongside the remote bar (e.g.
            ``"alignair-aa epochs"``).
        total_epochs: Total epochs the bar should advance through.
        disable: Force-disable even if Fernsicht is installed.
    """

    def __init__(self, desc: str = "training", total_epochs: Optional[int] = None,
                 disable: bool = False):
        super().__init__()
        self.desc = desc
        self.total_epochs = total_epochs
        self.disable = disable
        self._url = None

        if self.disable:
            return

        try:
            from fernsicht import manual as _manual
        except Exception as exc:  # noqa: BLE001
            logger.info("Fernsicht not available (%s); skipping remote progress URL.", exc)
            self.disable = True
            return

        bar = None
        try:
            bar = _manual(total=total_epochs, desc=desc, unit="ep")
        except Exception as exc:  # noqa: BLE001
            logger.warning("FernsichtCallback failed to start (%s); skipping URL.", exc)
            self.disable = True
            return

        self._url = getattr(bar, "url", None)

        # Tear down the WebRTC transport immediately. The room URL remains
        # valid for viewers; we just don't push live updates. This avoids
        # the asyncio/aiortc + TF training-thread race that segfaults
        # mid-fit on some configurations.
        try:
            close = getattr(bar, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001
            pass

        if self._url:
            banner = "=" * 70
            print(
                f"\n{banner}\n[fernsicht] monitor this run at: {self._url}\n{banner}\n",
                flush=True,
            )

    def on_train_begin(self, logs=None):
        # The URL was already printed in __init__; if it didn't print there,
        # there's no useful URL to surface here either.
        return

    def on_epoch_end(self, epoch, logs=None):
        # Live updates intentionally disabled — see module docstring.
        return

    def on_train_end(self, logs=None):
        return
