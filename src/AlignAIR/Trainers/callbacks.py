"""Custom Keras callbacks used by AlignAIR training."""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


# Fernsicht is an optional remote progress viewer
# (https://github.com/MuteJester/Fernsicht). It's imported lazily inside
# the callback because importing it after TensorFlow at module load time
# can race the asyncio / WebRTC event loop fernsicht starts and exit the
# process silently. The lazy import also keeps the optional dep truly
# optional — uninstalled is fine, --no_fernsicht is fine, no early load.


class FernsichtCallback(tf.keras.callbacks.Callback):
    """Stream per-epoch training metrics to a Fernsicht remote progress bar.

    Use to watch a long McCleary GPU run from anywhere (the Fernsicht
    relay handles NAT/firewall traversal). The callback degrades to a
    no-op if ``fernsicht`` isn't importable, so it's safe to include
    unconditionally.

    Args:
        desc: Short label shown alongside the remote bar
            (e.g. ``"alignair-aa epochs"``).
        total_epochs: Total epochs the bar should advance through.
            Defaults to whatever ``model.fit`` reports via the
            ``epochs`` key in ``on_train_begin``'s logs.
        disable: Force-disable even if Fernsicht is installed.
    """

    def __init__(self, desc: str = "training", total_epochs: Optional[int] = None,
                 disable: bool = False):
        super().__init__()
        self.desc = desc
        self.total_epochs = total_epochs
        self.disable = disable
        self._bar = None

    def on_train_begin(self, logs=None):
        if self.disable:
            return
        try:
            # Lazy import: fernsicht starts an asyncio/WebRTC loop on import
            # which has caused silent process exit when imported alongside
            # TensorFlow at module load. Importing here keeps everything
            # downstream of `import AlignAIR.API.TrainModel` safe.
            from fernsicht import manual as _manual
        except Exception as exc:  # noqa: BLE001
            logger.info("Fernsicht not available (%s); disabling remote progress bar.", exc)
            self.disable = True
            return

        total = self.total_epochs
        if total is None and self.params:
            total = self.params.get("epochs")
        try:
            self._bar = _manual(total=total, desc=self.desc, unit="ep")
        except Exception as exc:  # noqa: BLE001
            logger.warning("FernsichtCallback failed to start (%s); disabling.", exc)
            self._bar = None
            self.disable = True

    def on_epoch_end(self, epoch, logs=None):
        if self._bar is None:
            return
        logs = logs or {}
        # Keep the postfix small — the remote viewer truncates long strings.
        postfix = {
            k: float(v)
            for k, v in logs.items()
            if k in ("loss", "val_loss", "segmentation_loss",
                     "classification_loss", "v_allele_auc", "j_allele_auc")
            and isinstance(v, (int, float))
        }
        try:
            self._bar.update(1, **postfix)
        except Exception as exc:  # noqa: BLE001
            logger.debug("FernsichtCallback update failed (%s); continuing.", exc)

    def on_train_end(self, logs=None):
        if self._bar is None:
            return
        try:
            close = getattr(self._bar, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001
            pass
        self._bar = None
