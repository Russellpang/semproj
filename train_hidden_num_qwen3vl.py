"""
This script extracts image features from a pretrained Qwen3VL model and 
trains a probe classifier to predict the hidden number labels from these 
features. It uses PCA for dimensionality reduction and supports both logistic 
regression and linear SVM classifiers.
"""
import os
import json
import base64
import random
import joblib
import numpy as np
import torch
import argparse

from io import BytesIO
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

LABEL_TO_ID = {"X": 0, "O": 1, ".": 2}

def _decode_image_b64(image_b64):
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def str_to_label_ids(s):
    y = np.empty((36,), dtype=np.int64)
    for i, ch in enumerate(s):
        y[i] = LABEL_TO_ID[ch]
    return y

def build_labels(samples):
    N = len(samples)
    y_token = np.empty((N, 36), dtype=np.int64)
    for i, d in enumerate(samples):
        y_token[i] = str_to_label_ids(d["str_c"])
    return y_token

def load_samples(data_path):
    samples = []
    indices = []

    with open(data_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            """
            we only use partial data for training, the rest is used for testing, ensuring data balance
            """
            if (idx < 100) or (idx % 10 != 9):
                continue
            samples.append(json.loads(line))
            indices.append(idx)

    return samples, np.array(indices, dtype=np.int64)


class ImageOnlyDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        item = self.samples[i]
        image = _decode_image_b64(item["img_c_2"]["image_b64"])
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        return {"messages": messages}


class Collator:
    def __init__(self, processor, device, image_patch_size=16):
        self.processor = processor
        self.device = device
        self.image_patch_size = image_patch_size

    def __call__(self, batch):
        msgs = [b["messages"] for b in batch]
        image_inputs, _ = process_vision_info(msgs, image_patch_size=self.image_patch_size)
        vision_inputs = self.processor.image_processor(images=image_inputs, return_tensors="pt")
        pixel_values = vision_inputs["pixel_values"].to(self.device)
        image_grid_thw = vision_inputs["image_grid_thw"].to(self.device)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


@torch.no_grad()
def extract_image_features(model, processor, samples, batch_size, device, num_workers=0):
    ds = ImageOnlyDataset(samples)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=Collator(processor, device),
        pin_memory=False,
    )

    all_feats = []
    for batch in tqdm(loader):
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]

        out = model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        # HF versions may return tensor OR (tensor, extra)
        feats = out
        #print(feats)

        all_feats.append(feats[0][0].detach().float().cpu())

    #feats = torch.cat(all_feats, dim=0).numpy()  # [N,T,D]
    feats = np.stack(all_feats, axis=0)  # [N,T,D]
    print(f'feats have shape: {feats.shape}')
    return feats


def make_token_dataset(feats_NTD: np.ndarray, y_token: np.ndarray, case_idx: np.ndarray):
    X = feats_NTD[case_idx]
    y = y_token[case_idx]

    X = X.reshape(-1, X.shape[-1]).astype(np.float16)
    y = y.reshape(-1).astype(np.int64)
    return X, y


@dataclass
class Config:
    model_path: str
    data: str
    out_dir: str
    device: str = 'cuda'
    batch_size: int = 1
    seed: int = 42
    pca_dim: int = 32
    classifier: str = "logreg"   # "logreg" or "linear_svm"
    max_iter: int = 100
    val_split: float = 0.1
    num_workers: int = 0


def main(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)

    samples, sampled_training_indices = load_samples(cfg.data)
    y_labels = build_labels(samples)

    print("Loading processor/model...")
    processor = AutoProcessor.from_pretrained(cfg.model_path, do_resize=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()

    feats = extract_image_features(
        model=model,
        processor=processor,
        samples=samples,
        batch_size=cfg.batch_size,
        device=device,
        num_workers=cfg.num_workers,
    )

    num_cases = feats.shape[0]
    local_all = np.arange(num_cases, dtype=np.int64)
    train_local, val_local = train_test_split(
        local_all, test_size=cfg.val_split, random_state=cfg.seed, shuffle=True
    )

    X_train, y_train = make_token_dataset(feats, y_labels, train_local)
    X_val,   y_val   = make_token_dataset(feats, y_labels, val_local)

    pca = PCA(n_components=cfg.pca_dim, random_state=cfg.seed)
    X_train_p = pca.fit_transform(X_train)
    X_val_p   = pca.transform(X_val)

    if cfg.classifier == "logreg":
        clf = LogisticRegression(max_iter=cfg.max_iter, solver="lbfgs")
    elif cfg.classifier == "linear_svm":
        clf = LinearSVC(max_iter=cfg.max_iter)

    clf.fit(X_train_p, y_train)
    val_acc = accuracy_score(y_val, clf.predict(X_val_p))
    print(f"get_image_features probe: val_acc={val_acc:.4f}")

    save_path = os.path.join(cfg.out_dir, "vision_get_image_features_probe.joblib")
    joblib.dump(
        {"pca": pca, "clf": clf},
        save_path,
    )
    print(f"Saved probe to: {save_path}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_path", type=str, required=True)
    arg_parser.add_argument("--data", type=str, default="./testing_data_qwen3vl_6x6_0_to_20.jsonl")
    arg_parser.add_argument("--out_dir", type=str, default="./qwen3vl/probe_output")
    args = arg_parser.parse_args()
    cfg = Config(
        model_path=args.model_path,
        data=args.data,
        out_dir=args.out_dir,
        device="cuda",
        batch_size=1,
        pca_dim=32,
        classifier="logreg",
    )
    main(cfg)