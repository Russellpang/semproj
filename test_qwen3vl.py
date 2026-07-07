"""
This script is designed to test the Qwen3VL model for conditional generation tasks. 
It reads input data from a JSONL file, processes it according to the specified modality (text-only or multimodal), 
and performs inference using the Qwen3VL model.
According to the paper we only do experiments with output_type as 'num', so always set output_type to 'num' in the code.
You can run the script with the following command:
python test_qwen3vl.py --output_type num --input_type multimodal
"""
import json
import re
import torch
import os
import base64
import argparse

from tqdm import tqdm
from PIL import Image
from io import BytesIO 
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader, IterableDataset


class JsonlDataset(IterableDataset):
    def __init__(self, output_type, path='./testing_data_qwen3vl_6x6_0_to_20.jsonl', stride=1, modality='multimodal'):
        self.output_type = output_type
        self.path = path
        self.stride = stride
        self.modality = modality

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx % self.stride != 0:
                    continue
                d = json.loads(line)
                
                question_token = d["question"][f"{self.modality}_{self.output_type}"]
                system_prompt = d['system_prompt']
                if self.modality == 'multimodal':
                    image_b64 = d["img_c_2"]["image_b64"]       #This corresponds to the image with patch size 32*32, aligning with Qwen3VL's default vision configuration: 16 * 16 with spatial_merge_size 2. 
                    img_bytes2 = base64.b64decode(image_b64)
                    img = Image.open(BytesIO(img_bytes2)).convert("RGB")
                    messages = [
                        {
                            "role": "system",
                            "content": [
                            {"type": "text", "text": system_prompt},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": img,
                                },
                                {"type": "text", "text": question_token},
                            ],
                        }
                    ]
                else:
                    img = None
                    input_str = d["text_board"]
                    lst = list(input_str)
                    s = "[" + ",".join(f"'{x}'" for x in lst) + "], "
                    input_formatted = s + question_token
                    messages = [
                        {
                            "role": "system",
                            "content": [
                            {"type": "text", "text": system_prompt},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": input_formatted},
                            ],
                        }
                    ]

                yield {
                    "messages": messages,
                }


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        msgs = [item['messages'] for item in batch]
        inputs = [
            self.processor.apply_chat_template(m, tokenize=True,
                                               add_generation_prompt=True,
                                                return_dict=True,
                                                return_tensors="pt")
            for m in msgs
        ]
        #inputs = [i.to("cuda") for i in inputs]
        #image_inputs = [process_vision_info(msg)[0] for msg in msgs]

        # inputs = self.processor(
        #     text=texts,
        #     images=image_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )

        return {
            'inputs': inputs
        }

def batch_inference(model, processor, batch, log_file, batch_size):
    #inputs = batch["inputs"] ['input_ids'].to("cuda")
    # attention_mask = batch["inputs"] ['attention_mask'].to("cuda")
    # images = batch["inputs"] ['pixel_values'].to("cuda")
    inputs = batch["inputs"][0].to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=4096)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    FINAL_PAT = re.compile(r"FINAL_ANSWER\s*:\s*(.*)", re.IGNORECASE)

    if batch_size == 1:
        output_texts = processor.tokenizer.decode(
            generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        with open(log_file, "a") as f:
            output_with_white_space = output_texts
            matches = FINAL_PAT.findall(output_with_white_space)
            if not matches:
                suffix = ' FINAL_ANSWER: -1.'   #in case no final answer is generated after max_new_tokens, we assign -1 as the final answer, which is out of the normal answer range [0, 20] and can be easily identified.
            else:
                suffix = ''
            # last = matches[-1].strip()
            output = re.sub(r'(?<=\d)\s+(?=\d)', '', output_with_white_space)
            f.write(output + suffix + "\n")

    else:
        output_texts = processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        with open(log_file, "a") as f:
            for idx in range(len(output_texts)):
                output_with_white_space = output_texts[idx]
                matches = FINAL_PAT.findall(output_with_white_space)
                if not matches:
                    suffix = ' FINAL_ANSWER: -1.'
                else:
                    suffix = ''
                # last = matches[-1].strip()
                output = re.sub(r'(?<=\d)\s+(?=\d)', '', output_with_white_space)
                f.write(output + suffix + "\n")


def test_model(args):
    output_type = args.output_type
    input_type = args.input_type
    model_path = args.model_path
    data_path = args.data_path
    log_file = args.log_file
    stride = args.stride

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda')
    model.eval()
    # model.config.vision_config.spatial_merge_size = 1
    # model.config.vision_config.temporal_merge_size = 1
    
    test_dataset = JsonlDataset(output_type=output_type, path=data_path, stride=stride, modality=input_type)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Collator(processor),
        pin_memory=True,
        persistent_workers=False,
    )

    if log_file is None:
        log_file = f"./answer_history_qwen3vl_32B_{input_type}_{output_type}.log"

    if os.path.exists(log_file):
        os.remove(log_file)

    for _, batch in enumerate(tqdm(test_loader)):
        batch_inference(model, processor, batch, log_file, args.batch_size)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_type", choices=("num", "tf"), default='num', help="Output type: 'num' for number output, 'tf' for true/false output.")
    parser.add_argument("--input_type", choices=("text_only", "multimodal"), required=True, help="Modality of input text.")
    parser.add_argument("--model_path", type=str, default='Qwen/Qwen3-VL-32B-Instruct', help="Path to Qwen-VL model.")
    parser.add_argument("--data_path", type=str, default="./testing_data_qwen3vl_6x6_0_to_20.jsonl", help="Path to the testing data stored in JSONL format.")
    parser.add_argument("--log_file", type=str, help="Path to the log file.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_model(args)