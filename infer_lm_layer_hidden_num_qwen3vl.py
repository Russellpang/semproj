"""
Run inference with per-layer Qwen3VL LM hidden-state probes.

This loads the probes saved by train_lm_layer_hidden_num_qwen3vl.py, extracts
language-model hidden states at image-token positions, and reports count
accuracy for each layer probe.

Example:
python infer_lm_layer_hidden_num_qwen3vl.py \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --data ./testing_data_qwen3vl_6x6_0_to_20.jsonl \
  --probe_dir ./qwen3vl/lm_layer_probe_output
"""

import argparse
import base64
import glob
import json
import os
from io import BytesIO

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


BLACK_LABEL_ID = 0
IMAGE_TOKEN_ID = 151655


def _decode_image_b64(image_b64):
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def load_inference_samples(data_path):
    samples = []
    with open(data_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= 100 and idx % 10 == 9:
                continue
            item = json.loads(line)
            samples.append({"item": item})
    return samples


def build_messages(item):
    question = item["question"]["multimodal_num"]
    image = _decode_image_b64(item["img_c_2"]["image_b64"])
    messages = []
    if "system_prompt" in item:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": item["system_prompt"]}],
        })
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    })
    return messages


def encode_item(processor, item, device):
    inputs = processor.apply_chat_template(
        build_messages(item),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def image_token_positions(input_ids, image_token_id):
    positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    return positions


def load_probes(probe_dir):
    paths = sorted(glob.glob(os.path.join(probe_dir, "lm_hidden_state_*_probe.joblib")))

    probes = []
    for path in paths:
        probe = joblib.load(path)
        probe["probe_path"] = path
        probes.append(probe)

    probes.sort(key=lambda probe: int(probe["hidden_state_index"]))
    return probes


@torch.no_grad()
def extract_needed_hidden_states(model, inputs, layer_indices, image_token_id):
    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    positions = image_token_positions(inputs["input_ids"], image_token_id)
    return {
        layer_idx: outputs.hidden_states[layer_idx][0, positions, :].detach().float().cpu().numpy()
        for layer_idx in layer_indices
    }


def predict_black_count(probe, layer_features):
    pca_features = probe["pca"].transform(layer_features.astype(np.float32))
    pred_labels = probe["clf"].predict(pca_features)
    return int((pred_labels == BLACK_LABEL_ID).sum())


def run(args):
    samples = load_inference_samples(args.data)

    probes = load_probes(args.probe_dir)
    layer_indices = [int(probe["hidden_state_index"]) for probe in probes]

    device = torch.device(args.device)
    print(f"Loaded {len(samples)} samples")
    print(f"Loaded {len(probes)} layer probes from {args.probe_dir}")
    print("Loading processor/model...")
    processor = AutoProcessor.from_pretrained(args.model_path, do_resize=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    image_token_id = IMAGE_TOKEN_ID
    stats = {
        layer_idx: {"correct": 0, "total": 0}
        for layer_idx in layer_indices
    }

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as out_handle:
        for sample in tqdm(samples, desc="layer probe inference"):
            item = sample["item"]
            num_black = int(item["num_black"])
            inputs = encode_item(processor, item, device)
            features_by_layer = extract_needed_hidden_states(
                model=model,
                inputs=inputs,
                layer_indices=layer_indices,
                image_token_id=image_token_id,
            )

            predictions = {}
            for probe in probes:
                layer_idx = int(probe["hidden_state_index"])
                pred_count = predict_black_count(probe, features_by_layer[layer_idx])
                predictions[str(layer_idx)] = pred_count
                stats[layer_idx]["correct"] += int(pred_count == num_black)
                stats[layer_idx]["total"] += 1

            out_handle.write(json.dumps({
                "global_idx": item.get("global_idx"),
                "num_black": num_black,
                "pred_black_count_by_hidden_state": predictions,
            }) + "\n")

    for layer_idx in layer_indices:
        correct = stats[layer_idx]["correct"]
        total = stats[layer_idx]["total"]
        print((layer_idx, correct / total))
    print(f"Wrote per-sample predictions to: {args.out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Qwen3VL LM-layer hidden-number probes.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="./testing_data_qwen3vl_6x6_0_to_20.jsonl")
    parser.add_argument("--probe_dir", type=str, default="./qwen3vl/lm_layer_probe_output")
    parser.add_argument("--out_file", type=str, default="./qwen3vl/lm_layer_probe_output/rest_lm_layer_probe_predictions.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
