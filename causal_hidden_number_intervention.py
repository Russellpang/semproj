"""
Accuracy check for causal hidden-number interventions.

For each in-domain sample where num_black is in [0, 49], this script:

1. extracts vision-token latent representations,
2. uses the patch probe to identify black-stone tokens,
3. randomly disables k probe-positive image token indices for k in [1, --max_mask_count],
4. reports accuracy for each k over valid samples.

Example:
python causal_hidden_number_intervention.py \
  --model_path ./custom-model-finetune-num-vision-50-text-100 \
  --head_path ./prediction_head_vit.joblib \
  --data_path ./testing_data_formal.jsonl \
  --output_path ./causal_hidden_number_accuracy.json
"""

import argparse
import base64
import json
import os
import re
from io import BytesIO

import joblib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from model import CustomVLMProcessor, SimpleImageProcessor, VisionLanguageModel
from tokenizer import create_tokenizer

BLACK_LABEL_ID = 0


def _decode_image_b64(image_b64):
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


class InterventionDataset(IterableDataset):
    def __init__(self, data_path, stride):
        self.data_path = data_path
        self.stride = stride

    def __iter__(self):
        with open(self.data_path, "r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if line_idx % self.stride != 0:
                    continue

                item = json.loads(line)
                num_black = int(item["num_black"])
                if not 0 <= num_black <= 49:
                    continue

                yield {
                    "question_token": item["question"]["multimodal_num"],
                    "image_b64": item["img_c"]["image_b64"],
                    "num_black": num_black,
                }


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        messages = [[{"from": "human", "value": []}] for _ in batch]
        texts = [
            self.processor.apply_chat_template(
                message,
                item["question_token"],
                tokenize=False,
                add_generation_prompt=False,
                inference=True,
            )
            for message, item in zip(messages, batch)
        ]
        images = [_decode_image_b64(item["image_b64"]) for item in batch]
        encoded = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": encoded["pixel_values"],
            "num_blacks": [item["num_black"] for item in batch],
        }


@torch.no_grad()
def get_image_embeds(model, pixel_values):
    return model.vision_encoder(pixel_values)


def probe_black_indices(clf, image_embeds):
    features = image_embeds.detach().float().cpu().numpy()
    batch_size, num_tokens, hidden_size = features.shape
    features = features.reshape(batch_size * num_tokens, hidden_size)
    labels = clf.predict(features)
    labels = labels.reshape(batch_size, num_tokens)
    return [
        np.flatnonzero(row == BLACK_LABEL_ID).astype(np.int64).tolist()
        for row in labels
    ]


@torch.no_grad()
def generate_counts(model, processor, input_ids, attention_mask, pixel_values, disable_indices, max_new_tokens):
    output_ids = model.generate(
        input_ids=input_ids,
        images=pixel_values,
        attention_mask=attention_mask,
        disable_indices=disable_indices,
        max_new_tokens=max_new_tokens,
    )
    answer_ids = output_ids[:, input_ids.shape[1]:]
    answers = processor.tokenizer.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answers = [re.sub(r"\s+", "", answer) for answer in answers]
    return [int(answer) if answer else 0 for answer in answers]


def run(args):
    device = torch.device(args.device)

    print(f"Loading model from {args.model_path}")
    model = VisionLanguageModel.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
    ).to(device).eval()

    print("Loading tokenizer, processor, and probe")
    tokenizer = create_tokenizer(args.tokenizer_path, args.tokenizer_data_path)
    processor = CustomVLMProcessor(SimpleImageProcessor(), tokenizer)
    clf = joblib.load(args.head_path)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    stats = {
        k: {"correct": 0, "total": 0}
        for k in range(1, args.max_mask_count + 1)
    }

    print(f"Loading in-domain samples from {args.data_path}")
    dataset = InterventionDataset(
        data_path=args.data_path,
        stride=args.stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=Collator(processor),
        pin_memory=True,
        persistent_workers=False,
    )

    for batch in tqdm(loader, desc="Intervening"):
        ground_truths = batch["num_blacks"]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        image_embeds = get_image_embeds(model, pixel_values)
        black_indices_by_row = probe_black_indices(clf, image_embeds)

        for k in range(1, args.max_mask_count + 1):
            valid_rows = [row_idx
                for row_idx, (ground_truth, black_indices) in enumerate(
                    zip(ground_truths, black_indices_by_row)
                )
                if ground_truth >= k and len(black_indices) >= k
            ]
            if not valid_rows:
                continue

            disable_indices_by_row = [
                rng.choice(
                    black_indices_by_row[row_idx],
                    size=k,
                    replace=False,
                ).astype(np.int64).tolist()
                for row_idx in valid_rows
            ]
            row_tensor = torch.as_tensor(valid_rows, device=device, dtype=torch.long)
            masked_counts = generate_counts(
                model=model,
                processor=processor,
                input_ids=input_ids.index_select(0, row_tensor),
                attention_mask=attention_mask.index_select(0, row_tensor),
                pixel_values=pixel_values.index_select(0, row_tensor),
                disable_indices=disable_indices_by_row,
                max_new_tokens=args.max_new_tokens,
            )
            for row_idx, masked_count in zip(valid_rows, masked_counts):
                stats[k]["total"] += 1
                stats[k]["correct"] += int(masked_count == ground_truths[row_idx] - k)

    results = []
    for k, row in stats.items():
        accuracy = row["correct"] / row["total"] if row["total"] else 0.0
        result = {
            "masking_number": k,
            "correct": row["correct"],
            "total": row["total"],
            "accuracy": accuracy,
        }
        results.append(result)
        print(f"{k}: {row['correct']} / {row['total']} = {accuracy:.4f}")

    with open(args.output_path, "w", encoding="utf-8") as out:
        json.dump(results, out, indent=2)
        out.write("\n")
    print(f"Saved accuracy results to {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned num-output VLM.")
    parser.add_argument("--head_path", required=True, help="Path to prediction_head_vit.joblib.")
    parser.add_argument("--data_path", default="./testing_data_formal.jsonl")
    parser.add_argument("--output_path", default="./causal_hidden_number_accuracy.json")
    parser.add_argument("--tokenizer_path", default="./tokenizer_formal.json")
    parser.add_argument("--tokenizer_data_path", default="./training_data_formal.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_mask_count", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    run(parse_args())
