"""
This script tests the performance of a vision-language model on the textual and visual counting task. 
It loads a pretrained model, processes the test data, and logs the generated outputs for evaluation. 
The script supports both text-only and multimodal evaluation modes.
You can run the code in the following way:
python test_model.py --mode multimodal --output_type num --model_type finetune --model_path './custom-model-finetune-num-vision-50-text-100'
"""
import argparse
import base64
import json
import os
import re
from io import BytesIO

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from model import CustomVLMProcessor, SimpleImageProcessor, VisionLanguageModel
from tokenizer import create_tokenizer

class TestingDataset(IterableDataset):
    def __init__(self, mode, output_type, data_path, stride = 1):
        self.mode = mode
        self.output_type = output_type
        self.data_path = data_path
        self.stride = stride

    def __iter__(self):
        for p in self.data_path:
            with open(p, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx % self.stride != 0:
                        continue

                    data = json.loads(line)
                    if self.mode == "text":
                        yield from self._iter_text_samples(data)
                    else:
                        yield from self._iter_multimodal_samples(data)

    def _iter_text_samples(self, data):
        input_str = [" ".join(data["text_c"])]
        question_key = f"text_only_{self.output_type}"

        if self.output_type == "tf":
            for ref_key in ("text_eq", "text_ie"):
                yield {
                    "messages": input_str + [" ".join(data[ref_key])],
                    "question_token": data["question"][question_key],
                    "global_id": data["global_idx"],
                    "num_black": data["num_black"],
                    "groundtruth": True if ref_key == "text_eq" else False,
                }
            
        else:
            yield {
                "messages": input_str,
                "question_token": data["question"][question_key],
                "global_id": data["global_idx"],
                "num_black": data["num_black"],
                "groundtruth": data["num_black"],
            }

    def _iter_multimodal_samples(self, data):
        question_key = f"multimodal_{self.output_type}"
        
        if self.output_type == "tf":
            for ref_key in ("text_eq", "text_ie"):
                yield {
                    "question_token": data["question"][question_key],
                    "image_b64": data["img_c"]["image_b64"],
                    "messages": data[ref_key],
                    "global_id": data["global_idx"],
                    "num_black": data["num_black"],
                    "groundtruth": True if ref_key == "text_eq" else False,
                }
            
        else:
            yield {
                "question_token": data["question"][question_key],
                "image_b64": data["img_c"]["image_b64"],
                "messages": [],
                "global_id": data["global_idx"],
                "num_black": data["num_black"],
                "groundtruth": data["num_black"],
            }


class Collator:
    def __init__(self, processor, mode):
        self.processor = processor
        self.mode = mode

    def __call__(self, batch):
        if self.mode == "text":
            messages = [[{"from": "human", "value": item["messages"]}] for item in batch]
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
            encoded = self.processor(text=texts, return_tensors="pt", padding=True)
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "global_ids": [item["global_id"] for item in batch],
                "num_blacks": [item["num_black"] for item in batch],
                "groundtruths": [item["groundtruth"] for item in batch],
            }

        messages = []
        for item in batch:
            message_value = [" ".join(item["messages"])] if item["messages"] else []
            messages.append([{"from": "human", "value": message_value}])

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
        image_bytes = [base64.b64decode(item["image_b64"]) for item in batch]
        images = [Image.open(BytesIO(payload)).convert("RGB") for payload in image_bytes]
        encoded = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": encoded["pixel_values"],
            "global_ids": [item["global_id"] for item in batch],
            "num_blacks": [item["num_black"] for item in batch],
            "groundtruths": [item["groundtruth"] for item in batch],
        }


def batch_inference(model, processor, batch, log_file, max_new_tokens):
    input_ids = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    
    images = batch.get("pixel_values")
    if images is not None:
        images = images.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            images=images,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_texts = processor.tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    with open(log_file, "a", encoding="utf-8") as handle:
        for global_id, num_black, output_text, groundtruth in zip(
            batch["global_ids"], batch["num_blacks"], output_texts, batch["groundtruths"]
        ):
            normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", output_text)
            log_entry = {
                "global_id": global_id,
                "num_black": num_black,
                "generated_answer": normalized,
                "groundtruth": groundtruth,
            }
            handle.write(json.dumps(log_entry) + "\n")


def run_testing(args, model, processor):
    log_file = args.log_file

    dataset = TestingDataset(
        mode=args.mode,
        output_type=args.output_type,
        data_path=args.data_path,
        stride=args.stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=Collator(processor, args.mode),
        pin_memory=True,
        persistent_workers=False,
    )

    if os.path.exists(log_file):
        os.remove(log_file)

    for batch in tqdm(loader):
        batch_inference(model, processor, batch, log_file, args.max_new_tokens)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("text", "multimodal"), required=True, help="Testing mode: 'text' for text-only evaluation, 'multimodal' for vision-language evaluation.")
    parser.add_argument("--output_type", choices=("num", "tf"), required=True, help="Output type: 'num' for direct number output, 'tf' for true/false output.")
    parser.add_argument("--model_type", choices=("pretrained", "finetune"), required=True, help="Model type: 'pretrained' for using pre-trained model, 'finetune' for using the fine-tuned model.")
    parser.add_argument("--vision_boundary", type=int, default=49)
    parser.add_argument("--text_boundary", type=int, default=99)
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pretrained or fine-tuned model. If not provided, the script will look for a default path based on the model type output type, vision boundary, and text boundary.")
    parser.add_argument("--data_path", nargs="+", type=str, default=["./testing_data_formal.jsonl", './testing_data_formal_ood.jsonl'], help="Path to the testing data stored in JSONL format.")
    parser.add_argument("--log_file", type=str, help="Path to the log file.")
    parser.add_argument("--tokenizer_path", default="./tokenizer_formal.json")
    parser.add_argument("--tokenizer_data_path", default="./training_data_formal.jsonl")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.model_path is None:
        args.model_path = f"./custom-model-{args.model_type}-{args.output_type}-vision-{args.vision_boundary + 1}-text-{args.text_boundary + 1}"
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model path not provided and default path {args.model_path} does not exist.")
    
    if args.log_file is None:
        args.log_file = f"./answer-output-{args.model_type}-{args.mode}-{args.output_type}-vision-{args.vision_boundary + 1}-text-{args.text_boundary + 1}.log"
    
    print(f"Starting testing in {args.mode} mode with model: {args.model_path} whose output type is {args.output_type}")

    model = VisionLanguageModel.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = create_tokenizer(args.tokenizer_path, args.tokenizer_data_path)
    processor = CustomVLMProcessor(SimpleImageProcessor(), tokenizer)

    run_testing(args, model, processor)
    print(f"Finished testing. Outputs are logged in {args.log_file}")


if __name__ == "__main__":
    main()
