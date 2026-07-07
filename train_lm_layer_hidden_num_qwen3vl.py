"""
Train one hidden-number probe per Qwen3VL language-model layer.

This extracts Qwen3VL language-model hidden states at image-token positions,
trains a PCA + linear classifier probe for each selected hidden-state layer, and
saves one joblib file per layer.

Example:
python train_lm_layer_hidden_num_qwen3vl.py \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --data ./testing_data_qwen3vl_6x6_0_to_20.jsonl \
  --out_dir ./qwen3vl/lm_layer_probe_output
"""

import argparse
import base64
import json
import os
import random
from io import BytesIO
import joblib
import numpy as np
import torch

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


LABEL_TO_ID = {"X": 0, "O": 1, ".": 2}
IMAGE_TOKEN_ID = 151655

def _decode_image_b64(image_b64):
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def str_to_label_ids(board_str):
    labels = np.empty((36,), dtype=np.int64)
    for idx, ch in enumerate(board_str):
        labels[idx] = LABEL_TO_ID[ch]
    return labels


def load_probe_training_samples(data_path):
    samples = []
    with open(data_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < 100 or idx % 10 != 9:
                continue
            samples.append(json.loads(line))
    return samples


def build_messages(item):
    question = item["question"]["multimodal_num"]
    image = _decode_image_b64(item["img_c_2"]["image_b64"])
    messages = []
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


@torch.no_grad()
def extract_layer_features(model, processor, samples, layer_indices, image_token_id, device):
    features_by_layer = {layer_idx: [] for layer_idx in layer_indices}
    labels = []

    for item in tqdm(samples, desc="extract LM hidden states"):
        inputs = encode_item(processor, item, device)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        positions = image_token_positions(inputs["input_ids"], image_token_id)

        for layer_idx in layer_indices:
            layer_feats = outputs.hidden_states[layer_idx][0, positions, :]
            features_by_layer[layer_idx].append(layer_feats.detach().float().cpu().numpy())

        labels.append(str_to_label_ids(item["str_c"]))

    stacked = {
        layer_idx: np.stack(layer_feats, axis=0)
        for layer_idx, layer_feats in features_by_layer.items()
    }
    return stacked, np.stack(labels, axis=0)


def make_token_dataset(feats_ntd, labels_nt, case_indices):
    x = feats_ntd[case_indices].reshape(-1, feats_ntd.shape[-1]).astype(np.float32)
    y = labels_nt[case_indices].reshape(-1).astype(np.int64)
    return x, y


def train_probe_for_layer(feats_ntd, labels_nt, train_indices, val_indices, args):
    x_train, y_train = make_token_dataset(feats_ntd, labels_nt, train_indices)
    x_val, y_val = make_token_dataset(feats_ntd, labels_nt, val_indices)

    pca_dim = min(args.pca_dim, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    x_train_p = pca.fit_transform(x_train)
    x_val_p = pca.transform(x_val)

    if args.classifier == "logreg":
        clf = LogisticRegression(max_iter=args.max_iter, solver="lbfgs")
    elif args.classifier == "linear_svm":
        clf = LinearSVC(max_iter=args.max_iter)

    clf.fit(x_train_p, y_train)
    val_acc = accuracy_score(y_val, clf.predict(x_val_p))
    return pca, clf, val_acc


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    samples = load_probe_training_samples(args.data)

    print(f"Loaded {len(samples)} probe-training samples")
    print("Loading processor/model...")
    processor = AutoProcessor.from_pretrained(args.model_path, do_resize=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    image_token_id = IMAGE_TOKEN_ID
    inputs = encode_item(processor, samples[0], device)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    num_hidden_states = len(outputs.hidden_states)

    layer_indices = list(range(1, num_hidden_states))
    print(f"Training probes for hidden-state indices: {layer_indices}")

    feats_by_layer, labels = extract_layer_features(
        model=model,
        processor=processor,
        samples=samples,
        layer_indices=layer_indices,
        image_token_id=image_token_id,
        device=device,
    )

    case_indices = np.arange(labels.shape[0], dtype=np.int64)
    train_indices, val_indices = train_test_split(
        case_indices,
        test_size=args.val_split,
        random_state=args.seed,
        shuffle=True,
    )

    for layer_idx in layer_indices:
        print(f"Training layer hidden_state_index={layer_idx}")
        pca, clf, val_acc = train_probe_for_layer(
            feats_ntd=feats_by_layer[layer_idx],
            labels_nt=labels,
            train_indices=train_indices,
            val_indices=val_indices,
            args=args,
        )
        save_name = f"lm_hidden_state_{layer_idx:02d}_probe.joblib"
        save_path = os.path.join(args.out_dir, save_name)
        joblib.dump(
            {
                "pca": pca,
                "clf": clf,
                "hidden_state_index": layer_idx,
                "val_acc": val_acc,
            },
            save_path,
        )
        print(f"layer={layer_idx} val_acc={val_acc:.4f} saved={save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train one Qwen3VL LM-layer probe per hidden state.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, default="./testing_data_qwen3vl_6x6_0_to_20.jsonl")
    parser.add_argument("--out_dir", type=str, default="./qwen3vl/lm_layer_probe_output")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--classifier", choices=("logreg", "linear_svm"), default="logreg")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
