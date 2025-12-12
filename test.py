# debug_load_ad6.py
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# -------------------------------------------------------------------
# Custom layers used by AD6
# -------------------------------------------------------------------

class TimestepEmbedding(layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, t: tf.Tensor) -> tf.Tensor:
        half = self.dim // 2
        freqs = tf.exp(tf.range(half, dtype=tf.float32) * -np.log(10000.0) / (half - 1))
        args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        return emb

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim})
        return cfg


class SplitScaleShift(layers.Layer):
    """Avoid lambda deserialization issues in FiLM blocks."""
    def call(self, x):
        return tf.split(x, 2, axis=-1)

    def get_config(self):
        return super().get_config()


# -------------------------------------------------------------------
# AD6 candidate paths
# -------------------------------------------------------------------
AD6_PATH_CANDIDATES = [
    Path("data/models/ddpm_resunet_ad6_ema.keras"),
    Path("data/models/ddpm_resunetema.keras"),
    Path("data/models/ddpm_resunet_ad6.keras"),
]


def find_ad6_path():
    for p in AD6_PATH_CANDIDATES:
        if p.exists():
            print(f"[OK] Found AD6 candidate: {p}")
            return p
    raise FileNotFoundError(
        f"No AD6 model found. Checked: {AD6_PATH_CANDIDATES}"
    )


# -------------------------------------------------------------------
# Try loading AD6 without safe_mode
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== DEBUG: Trying to load AD6 model ===\n")
    print("TensorFlow version:", tf.__version__)

    path = find_ad6_path()
    print(f"\nLoading AD6 from:\n  {path.resolve()}\n")

    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={
                "TimestepEmbedding": TimestepEmbedding,
                "SplitScaleShift": SplitScaleShift,
            },
            safe_mode=False,   # <-- forced OFF
            compile=False,     # avoid loading optimizer/training losses
        )
        print("\n[✓] AD6 model loaded successfully!\n")
        print(model.summary())

    except Exception as e:
        import traceback
        print("\n[✗] Failed to load AD6 model!")
        print("Error:", e)
        traceback.print_exc()
