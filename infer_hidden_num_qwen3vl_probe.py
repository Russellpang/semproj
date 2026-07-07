"""
Run inference with the Qwen3VL hidden-number probe trained by test_hidden_num_qwen3vl.py.

Example:
python infer_hidden_num_qwen3vl_probe.py \
  --model_path Qwen/Qwen3-VL-32B-Instruct \
  --data ./testing_data_qwen3vl_6x6_0_to_20.jsonl \
  --probe_path ./qwen3vl/probe_output/vision_get_image_features_probe.joblib \
  --out_file ./qwen3vl/probe_output/rest_probe_predictions.jsonl
"""

import argparse
import base64
import json
import os
import joblib
import numpy as np
import torch

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


LABEL_TO_ID = {"X": 0, "O": 1, ".": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
BLACK_LABEL_ID = LABEL_TO_ID["X"]


def _decode_image_b64(image_b64):
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def str_to_label_ids(board_str):
    labels = np.empty((36,), dtype=np.int64)
    for idx, ch in enumerate(board_str):
        labels[idx] = LABEL_TO_ID[ch]
    return labels

def load_samples(data_path):
    samples = []
    with open(data_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx % 10 == 9 and idx >= 100:
                continue
            item = json.loads(line)
            samples.append({"item": item})
    return samples


class ImageOnlyDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]["item"]
        image = _decode_image_b64(item["img_c_2"]["image_b64"])
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        return {
            "messages": messages,
            "item": item,
        }


class Collator:
    def __init__(self, processor, device, image_patch_size=16):
        self.processor = processor
        self.device = device
        self.image_patch_size = image_patch_size

    def __call__(self, batch):
        messages = [sample["messages"] for sample in batch]
        image_inputs, _ = process_vision_info(messages, image_patch_size=self.image_patch_size)
        vision_inputs = self.processor.image_processor(images=image_inputs, return_tensors="pt")
        return {
            "pixel_values": vision_inputs["pixel_values"].to(self.device),
            "image_grid_thw": vision_inputs["image_grid_thw"].to(self.device),
            "items": [sample["item"] for sample in batch],
        }

@torch.no_grad()
def predict_batch(model, probe, batch):
    out = model.get_image_features(
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
    )

    feats = out
    while isinstance(feats, (tuple, list)):
        feats = feats[0]

    feats = feats.detach().float().cpu().numpy()

    if feats.ndim == 2:
        feats = feats[None, :, :]

    batch_size, num_tokens, hidden_dim = feats.shape

    flat_features = feats.reshape(batch_size * num_tokens, hidden_dim)
    pca_features = probe["pca"].transform(flat_features)
    pred_labels = probe["clf"].predict(pca_features).reshape(batch_size, num_tokens)
    return pred_labels


def evaluate_predictions(pred_labels, items):
    rows = []
    correct_counts = 0

    for pred, item in zip(pred_labels, items):
        num_black = int(item["num_black"])
        pred_black_count = int((pred == BLACK_LABEL_ID).sum())
        count_correct = pred_black_count == num_black

        correct_counts += int(count_correct)

        rows.append({
            "global_idx": item.get("global_idx"),
            "num_black": num_black,
            "pred_black_count": pred_black_count,
            "count_correct": count_correct,
        })

    return rows, correct_counts


def run(args):
    device = torch.device(args.device)
    samples = load_samples(args.data)

    print(f"Loaded {len(samples)} samples")
    print("Loading processor/model/probe...")
    processor = AutoProcessor.from_pretrained(args.model_path, do_resize=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    probe = joblib.load(args.probe_path)

    dataset = ImageOnlyDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Collator(processor, device, image_patch_size=args.image_patch_size),
        pin_memory=False,
    )

    total = 0
    count_correct = 0

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as out_handle:
        for batch in tqdm(loader, desc="probe inference"):
            pred_labels = predict_batch(model, probe, batch)
            rows, batch_correct = evaluate_predictions(
                pred_labels=pred_labels,
                items=batch["items"],
            )

            for row in rows:
                out_handle.write(json.dumps(row) + "\n")

            total += len(rows)
            count_correct += batch_correct

    print(f"Wrote predictions to: {args.out_file}")
    print(f"count_accuracy={count_correct / total:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with saved Qwen3VL hidden-number probe.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="./testing_data_qwen3vl_6x6_0_to_20.jsonl")
    parser.add_argument(
        "--probe_path",
        type=str,
        default="./qwen3vl/probe_output/vision_get_image_features_probe.joblib",
    )
    parser.add_argument("--out_file", type=str, default="./qwen3vl/probe_output/rest_probe_predictions.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_patch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
