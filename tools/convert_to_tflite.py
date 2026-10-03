"""
One-time conversion: Keras .h5 (ResNet50, ~226 MB, needs full TensorFlow)
  -> TFLite float16 (~48 MB, runs on the tiny `ai-edge-litert` runtime)
  -> head weights .npz (for exact NumPy Grad-CAM without TensorFlow)

Run locally (where TensorFlow is installed):
    python tools/convert_to_tflite.py
"""
import os
import sys
import json
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_CANDIDATES = [
    os.path.join(ROOT, "app", "pneumonia_resnet_best_model_1.h5"),
    os.path.join(ROOT, "pneumonia_resnet_best_model_1.h5"),
]
OUT_DIR = os.path.join(ROOT, "app", "models")
TFLITE_PATH = os.path.join(OUT_DIR, "pneumonia_classifier_fp16.tflite")
HEAD_PATH = os.path.join(OUT_DIR, "pneumonia_head.npz")
META_PATH = os.path.join(OUT_DIR, "pneumonia_meta.json")


def main():
    h5 = next((p for p in H5_CANDIDATES if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024), None)
    if not h5:
        sys.exit("Could not find pneumonia_resnet_best_model_1.h5")
    print(f"Loading {h5}")
    model = tf.keras.models.load_model(h5, compile=False)

    gap_idx = next(i for i, l in enumerate(model.layers) if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    conv_layer = model.layers[gap_idx - 1]
    dense_layers = [l for l in model.layers[gap_idx + 1:] if isinstance(l, tf.keras.layers.Dense)]
    print("Feature layer:", conv_layer.name, conv_layer.output.shape)
    print("Dense head:", [(d.name, d.units, d.activation.__name__) for d in dense_layers])

    # Two-output model: last conv feature map (for Grad-CAM) + final probability
    dual = tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="input")])
    def serve(x):
        feats, prob = dual(x, training=False)
        return {"features": feats, "probability": prob}

    concrete = serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], dual)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_bytes)
    print(f"Wrote {TFLITE_PATH} ({len(tflite_bytes) / 1e6:.1f} MB)")

    head = {}
    acts = []
    for i, d in enumerate(dense_layers):
        w, b = d.get_weights()
        head[f"W{i}"] = w.astype(np.float32)
        head[f"b{i}"] = b.astype(np.float32)
        acts.append(d.activation.__name__)
    np.savez_compressed(HEAD_PATH, **head)
    with open(META_PATH, "w") as f:
        json.dump({
            "input_size": [224, 224],
            "feature_layer": conv_layer.name,
            "feature_shape": list(conv_layer.output.shape[1:]),
            "dense_activations": acts,
            "source_model": os.path.basename(h5),
        }, f, indent=2)
    print(f"Wrote {HEAD_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()
