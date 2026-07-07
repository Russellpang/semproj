"""
Inference script for the patch-level prediction head on ViT outputs. This script loads 
a pre-trained VLM, a trained classifier, and a dataset of images. It processes the images 
through the model to obtain patch embeddings, applies the prediction head to classify each patch, 
and counts the number of black patches in each image. 
Finally, it evaluates the accuracy of the predictions against the ground truth counts.
"""
import argparse
import base64
import json
import os, joblib
import numpy as np
import torch

from dataclasses import dataclass
from io import BytesIO
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from model import VisionLanguageModel, SimpleImageProcessor, CustomVLMProcessor
from tokenizer import create_tokenizer

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

def load_data(data_path):
    samples = []
    num_black = []
    for d in data_path:
        with open(d, "r") as f:
            for _, line in enumerate(f):
                line = line.strip()
                d = json.loads(line)
                samples.append(d)
                num_black.append(d['num_black'])
    return samples, num_black

class PatchDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = _decode_image_b64(item["img_c"]["image_b64"])

        pixel_values = self.processor.image_processor([image], return_tensors="pt")["pixel_values"][0]

        return {
            "pixel_values": pixel_values,
        }


def _collate_patch_batch(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)

    return {
        "pixel_values": pixel_values,
    }

@dataclass
class TestingConfig:
    model_path: str
    data_jsonl: str
    head_path: str
    batch_size: int = 512
    device: str = "cuda"


def test_prediction_head(cfg):
    device = torch.device(cfg.device)

    model = VisionLanguageModel.from_pretrained(
        cfg.model_path, torch_dtype="auto", device_map=None, trust_remote_code=True
    ).to(device).eval()

    clf = joblib.load(cfg.head_path)

    image_processor = SimpleImageProcessor()
    tokenizer = create_tokenizer('./tokenizer_formal.json', './training_data_formal.jsonl')
    processor = CustomVLMProcessor(image_processor, tokenizer)
    data, num_black = load_data(cfg.data_jsonl)

    dataset = PatchDataset(data, processor)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate_patch_batch,
    )

    all_black_counts = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)

            patch_embeds = model.vision_encoder(pixel_values)
            
            B, N, D = patch_embeds.shape

            X = patch_embeds.reshape(B * N, D).detach().float().cpu().numpy()

            y = clf.predict(X).reshape(B, N)

            black_counts = (y == 0).sum(axis=1)
            all_black_counts.append(black_counts)

    all_black_counts = np.concatenate(all_black_counts, axis=0)  # (num_images,)

    print("done")
    return all_black_counts, num_black


def parse_args():
    parser = argparse.ArgumentParser(description="Inference a patch-level prediction head on ViT outputs.")
    parser.add_argument("--model_path", type = str, required=True)
    parser.add_argument("--data_jsonl", nargs="+", type=str, default=["./testing_data_formal.jsonl", './testing_data_formal_ood.jsonl'])
    parser.add_argument("--head_path", type = str, required=True, help="The path should be given similar to the following format: ./prediction_head/prediction_head.joblib")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return TestingConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    all_black_counts, num_black = test_prediction_head(config)
    print('evaluating accuracy...')
    correct = 0
    for i, res in enumerate(all_black_counts):
        if res == num_black[i]:
             correct += 1
        else:
            print(f"Wrong answer at line {i+1}: predicted {res}, actual {num_black[i]}, difference {abs(res - num_black[i])}")
    print(f"Total number of tests: {len(all_black_counts)}")
    print(f"overall accuracy: {correct / len(all_black_counts):.3f}")