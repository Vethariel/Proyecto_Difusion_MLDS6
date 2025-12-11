"""
AD6 - Conditional DDPM with residual U-Net, FiLM-style conditioning, cosine schedule, and EMA.

This experiment keeps the pixel-space setup of AD3/AD5 but strengthens the architecture:
- Residual Conv blocks with BatchNorm + SiLU
- Class conditioning via FiLM (scale/shift) merged with sinusoidal time embedding
- Optional classifier-free guidance during sampling (dropout of conditioning in training)
- Cosine noise schedule
- EMA weights for sampling stability
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# -----------------------------------------------------------------------------#
# Config
# -----------------------------------------------------------------------------#
DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
ARTIFACT_DIR = Path("artifacts_exp6")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SHAPE = (16, 16, 3)
T = 400  # more steps with cosine schedule
BATCH_SIZE = 256
EPOCHS = 60
LR = 2e-4
BASE_CHANNELS = 64
TIME_DIM = 128
CLASS_DIM = 64
DROP_PROB = 0.1  # classifier-free guidance drop probability

AUTOTUNE = tf.data.AUTOTUNE

# -----------------------------------------------------------------------------#
# Schedule (cosine, per Nichol & Dhariwal)
# -----------------------------------------------------------------------------#


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999).astype(np.float32)


betas = cosine_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)

betas_tf = tf.constant(betas, dtype=tf.float32)
alphas_tf = tf.constant(alphas, dtype=tf.float32)
alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=tf.float32)
sqrt_alphas_cumprod_tf = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)
sqrt_one_minus_alphas_cumprod_tf = tf.constant(np.sqrt(1 - alphas_cumprod), dtype=tf.float32)

# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#


def extract(a: tf.Tensor, t: tf.Tensor, shape: tf.TensorShape) -> tf.Tensor:
    out = tf.gather(a, t)
    reshape = (tf.shape(t)[0],) + (1,) * (len(shape) - 1)
    return tf.reshape(out, reshape)


class EMA:
    def __init__(self, model: tf.keras.Model, decay: float = 0.999):
        self.decay = decay
        self.shadow = [w.numpy() for w in model.weights]

    def update(self, model: tf.keras.Model):
        for i, w in enumerate(model.weights):
            self.shadow[i] = self.decay * self.shadow[i] + (1 - self.decay) * w.numpy()

    def apply_to(self, model: tf.keras.Model):
        for w, s in zip(model.weights, self.shadow):
            w.assign(s)


class TimestepEmbedding(layers.Layer):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def call(self, t: tf.Tensor) -> tf.Tensor:
        half = self.dim // 2
        freqs = tf.exp(tf.range(half, dtype=tf.float32) * -np.log(10000.0) / (half - 1))
        args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        return emb


def film_condition(cond: tf.Tensor, channels: int) -> Tuple[tf.Tensor, tf.Tensor]:
    h = layers.Dense(channels * 2)(cond)
    # Use Keras split to stay in symbolic graph
    scale, shift = layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(h)
    return layers.Reshape((1, 1, channels))(scale), layers.Reshape((1, 1, channels))(shift)


def residual_block(x: tf.Tensor, channels: int, cond: tf.Tensor, name: str) -> tf.Tensor:
    skip = x
    if x.shape[-1] != channels:
        skip = layers.Conv2D(channels, 1, padding="same", name=f"{name}_skip")(x)

    h = layers.Conv2D(channels, 3, padding="same", name=f"{name}_conv1")(x)
    h = layers.BatchNormalization(name=f"{name}_bn1")(h)
    h = tf.keras.activations.silu(h)

    h = layers.Conv2D(channels, 3, padding="same", name=f"{name}_conv2")(h)
    h = layers.BatchNormalization(name=f"{name}_bn2")(h)

    scale, shift = film_condition(cond, channels)
    h = h * (1 + scale) + shift
    h = tf.keras.activations.silu(h)
    return h + skip


def build_model(num_classes: int) -> tf.keras.Model:
    x_in = layers.Input(shape=IMG_SHAPE, name="x")
    t_in = layers.Input(shape=(), dtype=tf.int32, name="t")
    c_in = layers.Input(shape=(), dtype=tf.int32, name="class_id")

    # embeddings
    t_emb = TimestepEmbedding(TIME_DIM)(t_in)
    t_emb = layers.Dense(TIME_DIM, activation="silu")(t_emb)
    t_emb = layers.Dense(TIME_DIM, activation="silu")(t_emb)

    c_emb = layers.Embedding(num_classes + 1, CLASS_DIM)(c_in)  # +1 for null token (CFG)
    c_emb = layers.Dense(TIME_DIM, activation="silu")(c_emb)

    cond = layers.Concatenate()([t_emb, c_emb])
    cond = layers.Dense(TIME_DIM, activation="silu")(cond)

    # down
    h1 = residual_block(x_in, BASE_CHANNELS, cond, "down1a")
    h1 = residual_block(h1, BASE_CHANNELS, cond, "down1b")
    d1 = layers.MaxPooling2D(2)(h1)

    h2 = residual_block(d1, BASE_CHANNELS * 2, cond, "down2a")
    h2 = residual_block(h2, BASE_CHANNELS * 2, cond, "down2b")
    d2 = layers.MaxPooling2D(2)(h2)

    # bottleneck
    b = residual_block(d2, BASE_CHANNELS * 4, cond, "bottleneck1")
    b = residual_block(b, BASE_CHANNELS * 4, cond, "bottleneck2")

    # up
    u2 = layers.Conv2DTranspose(BASE_CHANNELS * 2, 2, strides=2, padding="same")(b)
    u2 = layers.Concatenate()([u2, h2])
    u2 = residual_block(u2, BASE_CHANNELS * 2, cond, "up2a")
    u2 = residual_block(u2, BASE_CHANNELS * 2, cond, "up2b")

    u1 = layers.Conv2DTranspose(BASE_CHANNELS, 2, strides=2, padding="same")(u2)
    u1 = layers.Concatenate()([u1, h1])
    u1 = residual_block(u1, BASE_CHANNELS, cond, "up1a")
    u1 = residual_block(u1, BASE_CHANNELS, cond, "up1b")

    out = layers.Conv2D(3, 3, padding="same", name="pred_noise")(u1)
    return models.Model([x_in, t_in, c_in], out, name="ddpm_resunet_cfg")


# -----------------------------------------------------------------------------#
# Data
# -----------------------------------------------------------------------------#


def load_data():
    data = np.load(DATA_PATH)
    images = data["images"].astype("float32")
    labels = data["labels"].astype("int32")
    return images, labels


def make_dataset(images: np.ndarray, labels: np.ndarray):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


# -----------------------------------------------------------------------------#
# Training & Sampling
# -----------------------------------------------------------------------------#


def q_sample(x0: tf.Tensor, t: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    return extract(sqrt_alphas_cumprod_tf, t, tf.shape(x0)) * x0 + extract(
        sqrt_one_minus_alphas_cumprod_tf, t, tf.shape(x0)
    ) * noise


def train():
    images, labels = load_data()
    num_classes = int(np.max(labels)) + 1
    ds = make_dataset(images, labels)

    model = build_model(num_classes)
    ema = EMA(model, decay=0.999)
    optimizer = optimizers.Adam(LR)

    @tf.function
    def train_step(x0, c_in):
        bsz = tf.shape(x0)[0]
        t = tf.random.uniform((bsz,), minval=0, maxval=T, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x0))
        x_t = q_sample(x0, t, noise)

        # classifier-free guidance: drop some labels to null token
        drop_mask = tf.random.uniform((bsz,)) < DROP_PROB
        c_train = tf.where(drop_mask, tf.cast(num_classes, tf.int32), c_in)

        with tf.GradientTape() as tape:
            noise_pred = model([x_t, t, c_train], training=True)
            loss = tf.reduce_mean(tf.square(noise - noise_pred))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    steps_per_epoch = len(images) // BATCH_SIZE
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for step, (x_batch, c_batch) in enumerate(ds):
            loss = train_step(x_batch, c_batch)
            losses.append(loss.numpy())
            ema.update(model)
        print(f"[AD6] Epoch {epoch:03d} | loss {np.mean(losses):.5f}")

    model.save(ARTIFACT_DIR / "ddpm_resunet_ad6.keras")
    ema.apply_to(model)
    model.save(ARTIFACT_DIR / "ddpm_resunet_ad6_ema.keras")
    np.savez(ARTIFACT_DIR / "schedule_ad6.npz", betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)
    print("Models and schedule saved to", ARTIFACT_DIR)


@tf.function
def predict_eps(model: tf.keras.Model, x: tf.Tensor, t: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
    return model([x, t, c], training=False)


def sample(
    model: tf.keras.Model,
    class_id: int,
    guidance_scale: float = 1.5,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
):
    if steps is None:
        steps = T
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    betas_s = betas[:steps]
    alphas_s = 1.0 - betas_s
    alphas_cumprod_s = np.cumprod(alphas_s)
    betas_tf_s = tf.constant(betas_s, dtype=tf.float32)
    alphas_tf_s = tf.constant(alphas_s, dtype=tf.float32)
    alphas_cumprod_tf_s = tf.constant(alphas_cumprod_s, dtype=tf.float32)

    x = tf.random.normal((1,) + IMG_SHAPE)
    class_batch = tf.fill((1,), tf.cast(class_id, tf.int32))
    null_batch = tf.fill((1,), tf.cast(model.layers[-1].input_shape[2][-1] - 1, tf.int32))

    for t_inv, t_val in enumerate(reversed(range(steps))):
        t_tensor = tf.fill((1,), tf.cast(t_val, tf.int32))

        eps_cond = predict_eps(model, x, t_tensor, class_batch)
        eps_uncond = predict_eps(model, x, t_tensor, null_batch)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        beta_t = betas_tf_s[t_val]
        alpha_t = alphas_tf_s[t_val]
        alpha_bar_t = alphas_cumprod_tf_s[t_val]

        mean = (1.0 / tf.sqrt(alpha_t)) * (x - beta_t * eps / tf.sqrt(1.0 - alpha_bar_t))
        if t_inv < steps - 1:
            noise = tf.random.normal(tf.shape(x))
            x = mean + tf.sqrt(beta_t) * noise
        else:
            x = mean

    x = tf.clip_by_value(x, 0.0, 1.0)
    return x.numpy()


if __name__ == "__main__":
    train()
