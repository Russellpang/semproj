"""
This script tests which attention heads are important for predicting the two digits of the count, 
by deactivating each head and checking if the prediction changes. 
It saves the resulting "keep" vectors (which heads are important for each digit) to a jsonl file.
"""

import torch
import re
import base64
import json
import os
import argparse

from PIL import Image
from io import BytesIO 
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, IterableDataset
from tokenizer import create_tokenizer
from model import CustomVLMProcessor, VisionLanguageModel, SimpleImageProcessor


class JsonlDataset(IterableDataset):
    def __init__(self, path='./testing_data_formal.jsonl', stride=1):
        self.path = path
        self.stride = stride        #speed-up testing on only a fractional of testcases

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx % self.stride != 0:
                    continue
                d = json.loads(line)
                ans = d['num_black']
                if (ans >=50):
                    continue
                
                yield {
                    "messages": [],
                    "question_token": d["question"][f"multimodal_num"],
                    "image_b64": d["img_c"]["image_b64"],
                    'answer': f"{ans:02d}"  
                }

class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        msgs = []
        for item in batch:
            if item["messages"]:
                msg_value = [" ".join(item["messages"])]
            else:
                msg_value = []
            m = [
                {"from": "human", "value": msg_value},
                {"from": "gpt",   "value": item["answer"]},
            ]
            msgs.append(m)

        questions = [item["question_token"] for item in batch]

        img_bytes2 = [base64.b64decode(item['image_b64']) for item in batch]
        images = [(Image.open(BytesIO(p))).convert("RGB") for p in img_bytes2]

        texts = [
            self.processor.apply_chat_template(
                m, q, tokenize=False,
                add_generation_prompt=False,
                inference=False
            )
            for m, q in zip(msgs, questions)
        ]

        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        assistant_start = self.processor.tokenizer.convert_tokens_to_ids(self.processor.assistant_start)
        eos      = self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.eos_token)
        pad_id   = self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.pad_token)

        T = input_ids.size(1)
        assistant_start_mask = (input_ids == assistant_start)
        eos_mask = (input_ids == eos)

        ass_pos = assistant_start_mask.float().argmax(dim=1)  # (B,)
        eos_pos = eos_mask.float().argmax(dim=1)  # (B,)

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        labels[pos <= ass_pos.unsqueeze(1)] = -100
        labels[pos >  eos_pos.unsqueeze(1)] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": enc["pixel_values"],
            "labels": labels,
        }

@torch.no_grad()
def tf_two_digit_preds(model, batch, head_mask):
    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels    = batch["labels"].to(device)
    pixel_values = batch["pixel_values"].to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        pixel_values=pixel_values,
        labels=None,
        head_mask=head_mask,
    )
    logits = out.logits  # (B,T,V)

    sup = (labels != -100)
    sup_cum = sup.long().cumsum(dim=1)
    p0 = (sup_cum == 1).float().argmax(dim=1)
    p1 = (sup_cum == 2).float().argmax(dim=1)

    B = input_ids.size(0)
    ar = torch.arange(B, device=device)

    gt1 = input_ids[ar, p0]
    gt2 = input_ids[ar, p1]

    pred1 = logits[ar, p0 - 1].argmax(dim=-1)
    pred2 = logits[ar, p1 - 1].argmax(dim=-1)

    return pred1, pred2, gt1, gt2, p0, p1


@torch.no_grad()
def per_digit_keep_masks(model, batch, L, H):
    device = next(model.parameters()).device

    base_mask = torch.ones(L, H, device=device)
    b1, b2, _, _, _, _ = tf_two_digit_preds(model, batch, base_mask)

    B = b1.size(0)
    keep = torch.zeros(B, 2, L, H, device=device, dtype=torch.int8)

    for l in range(L):
        for h in range(H):
            trial = base_mask.clone()
            trial[l, h] = 0
            p1, p2, _, _, _, _ = tf_two_digit_preds(model, batch, trial)

            keep[:, 0, l, h] = (p1 != b1).to(torch.int8)  # tens
            keep[:, 1, l, h] = (p2 != b2).to(torch.int8)  # units

    return keep

def run_testing(batch_size, num_workers, model, processor, stride, L, H):
    test_dataset = JsonlDataset(stride=stride)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=Collator(processor),
        pin_memory=True,
        persistent_workers=False,
    )

    out_path = f"./pruning_vectors_vision.jsonl"
    with open(out_path, "w") as f:
        for batch in tqdm(test_loader):
            keep = per_digit_keep_masks(model, batch, L, H)  # (B,2,L,H)
            keep_2lh = keep.squeeze(0)
            keep_np  = keep_2lh.detach().cpu().numpy()
            vec = keep_np.tolist()
            f.write(json.dumps({"keep": vec}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model. Output type should be num.")
    parser.add_argument("--stride", type=int, default=8, help="Stride for speeding up testing dataset.")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    model_path = args.model_path
    stride = args.stride
    batch_size = args.batch_size
    
    model = VisionLanguageModel.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    tokenizer = create_tokenizer('./tokenizer_formal.json', './training_data_formal.jsonl')
    vocab_size = len(tokenizer)
    processor = CustomVLMProcessor(SimpleImageProcessor(), tokenizer)
    L = model.decoder.num_blks
    H = model.decoder.num_heads
    run_testing(
        batch_size=batch_size, num_workers=0,
        model=model, processor=processor, stride=stride,
        L=L, H=H
    )
    print(f"Finished testing for model: {model_path} with stride: {stride}")
