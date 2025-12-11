"""
Compare AD3 (ddpm_conditional) vs AD6 (ddpm_resunet_ad6_ema) on the pixel-art dataset
using a PCA-based Feature-FID proxy per class.

Outputs a JSON with per-class scores and averages under artifacts_exp6/compare_ad3_ad6.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models

# -----------------------------------------------------------------------------#
# Paths and constants
# -----------------------------------------------------------------------------#
DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
MODEL_AD3 = Path("data/models/ddpm_conditional.keras")
MODEL_AD6 = Path("data/models/ddpm_resunet_ad6_ema.keras")  # use EMA for sampling
SCHEDULE_AD6 = Path("data/models/schedule_ad6.npz")
OUT_JSON = Path("artifacts_exp6/compare_ad3_ad6.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

IMG_SHAPE = (16, 16, 3)
T_AD3 = 300
BETA_START_AD3 = 1e-4
BETA_END_AD3 = 2e-2
SAMPLES_PER_CLASS = 200
REAL_PER_CLASS = 1000
PCA_COMPONENTS = 20
GUIDANCE_SCALE_AD6 = 1.5
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"


# -----------------------------------------------------------------------------#
# Shared utilities
# -----------------------------------------------------------------------------#
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
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


def compute_feature_fid(real_imgs: np.ndarray, gen_imgs: np.ndarray, n_components: int = PCA_COMPONENTS) -> float:
    real_flat = real_imgs.reshape(len(real_imgs), -1)
    gen_flat = gen_imgs.reshape(len(gen_imgs), -1)
    pca = PCA(n_components=n_components)
    real_f = pca.fit_transform(real_flat)
    gen_f = pca.transform(gen_flat)

    mu_real = real_f.mean(axis=0)
    mu_gen = gen_f.mean(axis=0)
    cov_real = np.cov(real_f, rowvar=False)
    cov_gen = np.cov(gen_f, rowvar=False)

    mean_dist = np.sum((mu_real - mu_gen) ** 2)
    cov_dist = np.sum((cov_real - cov_gen) ** 2)
    return float(mean_dist + cov_dist)


def get_classes(labels: np.ndarray) -> List[int]:
    return sorted(list(set(int(x) for x in np.unique(labels))))


# -----------------------------------------------------------------------------#
# AD3: load and sampling (linear betas)
# -----------------------------------------------------------------------------#
betas_ad3 = np.linspace(BETA_START_AD3, BETA_END_AD3, T_AD3, dtype=np.float32)
alphas_ad3 = 1.0 - betas_ad3
alphas_cumprod_ad3 = np.cumprod(alphas_ad3, axis=0)

betas_ad3_tf = tf.constant(betas_ad3, dtype=tf.float32)
alphas_ad3_tf = tf.constant(alphas_ad3, dtype=tf.float32)
alphas_cumprod_ad3_tf = tf.constant(alphas_cumprod_ad3, dtype=tf.float32)


def load_ad3() -> tf.keras.Model:
    return models.load_model(MODEL_AD3, custom_objects={"TimestepEmbedding": TimestepEmbedding})


def sample_ad3(model: tf.keras.Model, class_id: int, num_samples: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    with tf.device(DEVICE):
        x = tf.random.normal((num_samples,) + IMG_SHAPE)
        c = tf.fill((num_samples,), tf.cast(class_id, tf.int32))
        for t in reversed(range(T_AD3)):
            t_b = tf.fill((num_samples,), tf.cast(t, tf.int32))
            eps = model([x, t_b, c], training=False)
            beta_t = betas_ad3_tf[t]
            alpha_t = alphas_ad3_tf[t]
            alpha_bar_t = alphas_cumprod_ad3_tf[t]
            mean = (1.0 / tf.sqrt(alpha_t)) * (x - beta_t * eps / tf.sqrt(1.0 - alpha_bar_t))
            if t > 0:
                x = mean + tf.sqrt(beta_t) * tf.random.normal(tf.shape(x))
            else:
                x = mean
        x = tf.clip_by_value(x, 0.0, 1.0)
    return x.numpy()


# -----------------------------------------------------------------------------#
# AD6: load and sampling (cosine betas, guidance)
# -----------------------------------------------------------------------------#
schedule_ad6 = np.load(SCHEDULE_AD6)
betas_ad6 = schedule_ad6["betas"]
alphas_ad6 = schedule_ad6["alphas"]
alphas_cumprod_ad6 = schedule_ad6["alphas_cumprod"]


def load_ad6() -> tf.keras.Model:
    # AD6 uses Lambda layers; allow unsafe deserialization when loading a trusted local file.
    return models.load_model(
        MODEL_AD6,
        custom_objects={"TimestepEmbedding": TimestepEmbedding},
        safe_mode=False,
    )


def infer_null_index(model: tf.keras.Model) -> int:
    for lyr in model.layers:
        if isinstance(lyr, layers.Embedding):
            return int(lyr.input_dim) - 1
    raise ValueError("Embedding layer not found to infer null token.")


def sample_ad6(
    model: tf.keras.Model,
    class_id: int,
    num_samples: int,
    guidance_scale: float = GUIDANCE_SCALE_AD6,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    if steps is None:
        steps = len(betas_ad6)
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    betas_s = betas_ad6[:steps]
    alphas_s = alphas_ad6[:steps]
    alphas_cumprod_s = alphas_cumprod_ad6[:steps]

    betas_tf = tf.constant(betas_s, dtype=tf.float32)
    alphas_tf = tf.constant(alphas_s, dtype=tf.float32)
    alphas_cumprod_tf = tf.constant(alphas_cumprod_s, dtype=tf.float32)

    null_idx = infer_null_index(model)

    with tf.device(DEVICE):
        x = tf.random.normal((num_samples,) + IMG_SHAPE)
        c = tf.fill((num_samples,), tf.cast(class_id, tf.int32))
        c_null = tf.fill((num_samples,), tf.cast(null_idx, tf.int32))

        for t_inv, t_val in enumerate(reversed(range(steps))):
            t_b = tf.fill((num_samples,), tf.cast(t_val, tf.int32))
            eps_cond = model([x, t_b, c], training=False)
            eps_uncond = model([x, t_b, c_null], training=False)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            beta_t = betas_tf[t_val]
            alpha_t = alphas_tf[t_val]
            alpha_bar_t = alphas_cumprod_tf[t_val]

            mean = (1.0 / tf.sqrt(alpha_t)) * (x - beta_t * eps / tf.sqrt(1.0 - alpha_bar_t))
            if t_inv < steps - 1:
                x = mean + tf.sqrt(beta_t) * tf.random.normal(tf.shape(x))
            else:
                x = mean

        x = tf.clip_by_value(x, 0.0, 1.0)
    return x.numpy()


# -----------------------------------------------------------------------------#
# Evaluation loop
# -----------------------------------------------------------------------------#
def evaluate_model(
    model_name: str,
    sampler,
    real_images: np.ndarray,
    labels: np.ndarray,
    classes: List[int],
) -> Dict:
    results = {}
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        real_subset = real_images[np.random.choice(cls_indices, size=min(REAL_PER_CLASS, len(cls_indices)), replace=False)]
        gen = sampler(class_id=cls, num_samples=SAMPLES_PER_CLASS)
        fid = compute_feature_fid(real_subset, gen)
        results[int(cls)] = fid
        print(f"{model_name} | class {cls} | feature-FID {fid:.4f}")
    results["avg"] = float(np.mean(list(results.values())))
    return results


def main():
    data = np.load(DATA_PATH)
    images = data["images"].astype("float32")
    labels = data["labels"].astype("int32")
    classes = get_classes(labels)

    print("Loaded dataset:", images.shape, "classes:", classes)

    print("Loading AD3...")
    ad3 = load_ad3()
    print("Loading AD6 (EMA)...")
    ad6 = load_ad6()

    metrics = {}
    metrics["AD3"] = evaluate_model(
        "AD3",
        lambda class_id, num_samples: sample_ad3(ad3, class_id=class_id, num_samples=num_samples),
        images,
        labels,
        classes,
    )
    metrics["AD6"] = evaluate_model(
        "AD6",
        lambda class_id, num_samples: sample_ad6(ad6, class_id=class_id, num_samples=num_samples),
        images,
        labels,
        classes,
    )

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics to", OUT_JSON)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
