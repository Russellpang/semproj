"""
This script trains a patch-level prediction head on top of the ViT outputs 
of out VLM. It uses a dataset of images and corresponding labels to train a 
classifier (either logistic regression or linear SVM) that predicts the label 
of a specific patch in the image. The hidden number is determined by aggregating 
the predictions of the classifier across all patches in the image.
"""

import argparse
import base64
import json
import random
import os, joblib
import numpy as np
import torch

from dataclasses import dataclass
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from tokenizer import create_tokenizer
from model import VisionLanguageModel, SimpleImageProcessor, CustomVLMProcessor

LABEL_TO_ID = {
    "X": 0,  # black
    "O": 1,  # white
    ".": 2,  # empty
    "black": 0,
    "white": 1,
    "empty": 2,
    'c': 0,
    'b': 1,
    'a': 2,
}

def _decode_image_b64(image_b64):
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def load_samples_jsonl(cfg):
    data_path = cfg.data_path
    seed = cfg.seed
    vision_boundary = cfg.vision_boundary
    rng = random.Random(seed)

    samples = []
    with open(data_path, "r") as f:
        current_samples = []
        for idx, line in enumerate(f):
            line = line.strip()
            d = json.loads(line)

            if (d['num_black'] == 0 or d['num_black'] > vision_boundary): continue

            current_samples.append(d)
            if ((idx + 1) % cfg.num_data_point == 0):
                white_stone_symbol = "O"
                eligible = [s for s in current_samples if white_stone_symbol in s["str_c"]]
                samples.extend(rng.sample(eligible, k=100))
                current_samples = []

    return samples


def _board_str_to_indices(board_str):
    indices = {"black": [], "white": [], "empty": []}
    for idx, ch in enumerate(board_str):
        if ch in ("X"):
            indices["black"].append(idx)
        elif ch in ("O"):
            indices["white"].append(idx)
        else:
            indices["empty"].append(idx)
    return indices


def build_patch_samples(entries, seed):
    rng = random.Random(seed)
    samples = []

    for idx, item in enumerate(entries):
        board_str = item['str_c']
        indices = _board_str_to_indices(board_str)

        def add_sample(item, label, candidates, samples):
            samples.append(
                {
                    "entry": item,
                    "patch_index": rng.choice(candidates),
                    "label": label,
                }
            )

        """
        Here we balance the number of samples for each label, so that we have 
        number of "black" samples : number of "rest" samples = 1:1, while in the
        "rest" samples, we have number of "white" samples : number of "empty" 
        samples = 1:1.
        """
        if idx % 2 == 0:
            add_sample(item, "black", indices["black"], samples)
            add_sample(item, "white", indices["white"], samples)
        else:
            add_sample(item, "black", indices["black"], samples)
            add_sample(item, "empty", indices["empty"], samples)

    return samples


class PatchLabelDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        data = item['entry']

        image = _decode_image_b64(data["img_c"]["image_b64"])

        label_id = LABEL_TO_ID[item["label"]]
        patch_index = item["patch_index"]

        pixel_values = self.processor.image_processor([image], return_tensors="pt")["pixel_values"][0]

        return {
            "pixel_values": pixel_values,
            "patch_index": patch_index,
            "label": label_id,
        }


def _collate_patch_batch(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    patch_index = [b["patch_index"] for b in batch]
    patch_index = torch.tensor(patch_index, dtype=torch.long)

    return {
        "pixel_values": pixel_values,
        "patch_index": patch_index,
        "labels": labels,
    }

def get_patch_embeddings(patch_embeds, patch_index):
    batch_indices = torch.arange(patch_embeds.size(0), device=patch_embeds.device)
    return patch_embeds[batch_indices, patch_index]

@dataclass
class TrainConfig:
    model_path: str
    vision_boundary: int = 49
    data_path: str = './training_data_formal.jsonl'
    output_dir: str = './'
    batch_size: int = 512
    device: str = "cuda"
    classifier: str = "logreg"
    max_iter: int = 50
    val_split: float = 0.1
    num_data_point: int = 8192
    seed: int = 42


def train_prediction_head(cfg):
    device = torch.device(cfg.device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("Loading model...")
    model = VisionLanguageModel.from_pretrained(
        cfg.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    model.to(device).eval()

    image_processor = SimpleImageProcessor()
    tokenizer = create_tokenizer('./tokenizer_formal.json', './training_data_formal.jsonl')
    processor = CustomVLMProcessor(image_processor, tokenizer)

    print("Loading data...")
    entries = load_samples_jsonl(cfg)
    print('building patch samples...')
    patch_samples = build_patch_samples(entries, cfg.seed)

    print('splitting train/val...')
    train_samples, val_samples = train_test_split(
        patch_samples,
        test_size=cfg.val_split,
        random_state=cfg.seed,
        shuffle=True,
    )

    def extract_features(model, dataset):
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate_patch_batch)
        features = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device)
                #input_ids = batch['input_ids'].to("cuda")
                #attention_mask = batch['attention_mask'].to("cuda")
                patch_index = batch["patch_index"].to(device)
                batch_labels = batch["labels"].cpu().numpy()

                patch_embeds = model.vision_encoder(pixel_values)
                #out = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=None)
                #x0 = model.decoder._cache_after_block0
                #patch_embeds = x0[:, 3:364, :]

                selected = get_patch_embeddings(patch_embeds, patch_index)
                features.append(selected.detach().cpu().numpy())
                labels.append(batch_labels)
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    print("Extracting features...")
    X_train, y_train = extract_features(model,PatchLabelDataset(train_samples, processor))
    X_val, y_val = extract_features(model, PatchLabelDataset(val_samples, processor))

    if cfg.classifier == "logreg":
        clf = LogisticRegression(max_iter=cfg.max_iter, multi_class="auto")
    elif cfg.classifier == "linear_svm":
        clf = LinearSVC(max_iter=cfg.max_iter)

    print("Training classifier...")
    clf.fit(X_train, y_train)

    print("Evaluating on validation set...")
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"val accuracy: {val_acc:.4f}")

    print("Saving classifier...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(cfg.output_dir, f"prediction_head_vit.joblib"))

def parse_args():
    parser = argparse.ArgumentParser(description="Train a patch-level prediction head on ViT outputs.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vision_boundary", default=49, help='The vision boundary corresponding to the model.')
    parser.add_argument("--data_path", default='./training_data_formal.jsonl', help='Data for training this classifier.')
    parser.add_argument("--output_dir", type=str, default='./')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier", choices=["logreg", "linear_svm"], default="logreg")
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_data_point", type=int, default=8192, help='The number of training samples generated for each counting number.')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train_prediction_head(config)
