"""
This script generates data in the required format for both text-only pretraining and multimodal finetuning.
You can run the code in the following way: 
python generate_training_data_json.py --process_type finetune --output_type num
"""
import os
import json
import random
import base64
import argparse

from pathlib import Path
from tqdm import tqdm
from PIL import Image 
from io import BytesIO 

def merge_blockwise(a, b, block_size):
    """
    Merge blocks according to the relative sizes of a and b.
    """

    if not a:
        return b
    if not b:
        return a

    out = []
    a_index = 0
    b_index = 0
    if len(b) >= len(a):
        a_blocks_per_cycle = 1
        b_blocks_per_cycle = max(1, round(len(b) / len(a)))
    else:
        a_blocks_per_cycle = max(1, round(len(a) / len(b)))
        b_blocks_per_cycle = 1

    while a_index < len(a) or b_index < len(b):
        for _ in range(a_blocks_per_cycle):
            if a_index < len(a):
                out.extend(a[a_index:a_index + block_size])
                a_index += block_size

        for _ in range(b_blocks_per_cycle):
            if b_index < len(b):
                out.extend(b[b_index:b_index + block_size])
                b_index += block_size

    return out

def generate_json_annotations(process_type, output_type, vision_boundary, text_boundary, data_path, output_dir):
    if process_type == 'pretrain':
        filename = f'training_19x19_text_{process_type}_output_{output_type}_vision_{vision_boundary}_text_{text_boundary}.jsonl'
    else: 
        filename = f'training_19x19_multimodal_{process_type}_output_{output_type}_vision_{vision_boundary}_text_{text_boundary}.jsonl'
    
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    out_file = os.path.join(output_dir, filename)
    if os.path.exists(out_file):
        os.remove(out_file)

    annotations = []
    text_data = [] # now we add also pre-trained text data to avoid forgetting

    if process_type == 'pretrain' and output_type == 'tf':
        for i, d in tqdm(enumerate(data)):
            #-----this is for text pretraining true false data generation-----
            num_black = d['num_black']
            if num_black > text_boundary:
                break

            text_c = d['text_c']
            text_eq = d['text_eq']
            text_ie = d['text_ie']
            given_string_text_c = " ".join(text_c)
            given_string_text_eq = " ".join(text_eq)
            given_string_text_ie = " ".join(text_ie)

            question = d['question']['text_only_tf']
            ref_string = given_string_text_eq if i % 2 == 0 else given_string_text_ie
            label = 'True' if i % 2 == 0 else 'False'

            ann = {
                "img_b64": None,
                "question_token": question,
                "conversations": [
                    {"from": "human", "value": [given_string_text_c, ref_string]},
                    {"from": "gpt", "value": label}
                ]
            }
            annotations.append(ann)

        random.shuffle(annotations)
        with open(out_file, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        print(f"Generated {len(annotations)} annotations and saved to {out_file}")
        
    elif process_type == 'pretrain' and output_type == 'num':
        for i, d in tqdm(enumerate(data)):
            #-----this is for text pretraining number output data generation-----
            num_black = d['num_black']
            if num_black > text_boundary:
                break

            text_c = d['text_c']
            given_string_text_c = " ".join(text_c)
            question = d['question']['text_only_num']
            ans = d['num_black']

            ann = {
                "img_b64": None,
                "question_token": question,
                "conversations": [
                    {"from": "human", "value": [given_string_text_c]},
                    {"from": "gpt", "value": f"{ans:02d}"}
                ]
            }
            annotations.append(ann)

        random.shuffle(annotations)
        with open(out_file, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        print(f"Generated {len(annotations)} annotations and saved to {out_file}")
    
    elif process_type == 'finetune' and output_type == 'tf':
        for i, d in tqdm(enumerate(data)):

            input_str_c = d['text_c']
            input_str_eq = d['text_eq']
            input_str_ie = d['text_ie']
            given_string_c = " ".join(input_str_c)
            given_string_eq = " ".join(input_str_eq)
            given_string_ie = " ".join(input_str_ie)

            #----- text pretraining true false data to avoid catastrophic forgetting-----
            if (d['num_black'] <= text_boundary and (i % 4 == 0 or i % 4 == 2)):
                tf_label = 'True' if (i % 4 == 0) else 'False'
                ref_string = given_string_eq if tf_label == 'True' else given_string_ie

                question_text = d['question']['text_only_tf']
                ann = {
                    "img_b64": None,
                    "question_token": question_text,
                    "conversations": [
                        {"from": "human", "value": [given_string_c, ref_string]},
                        {"from": "gpt", "value": tf_label}
                    ]
                }

                text_data.append(ann)

            #-----this is for finetuning img tf generation-----
            if (d['num_black'] > vision_boundary):
                continue

            img_b64 = d['img_c']['image_b64']
            # img_bytes2 = base64.b64decode(d['img_c']['image_b64']) 
            # img2 = Image.open(BytesIO(img_bytes2))
            question_img = d['question']['multimodal_tf']
            ref_string = given_string_eq if i % 2 == 0 else given_string_ie
            label = 'True' if i % 2 == 0 else 'False'
            
            ann = {
                "img_b64": img_b64,
                "question_token": question_img,
                "conversations": [
                    {"from": "human", "value": [ref_string]},
                    {"from": "gpt", "value": label}
                ]
            }
            annotations.append(ann)
            
        
        random.shuffle(annotations)
        random.shuffle(text_data)
        print(f'length of pre-trained text data: {len(text_data)}')
        print(f'length of finetuned vision data: {len(annotations)}')
        
        final = merge_blockwise(text_data, annotations, 2048)
        # final = annotations
        print(f'length of final finetuning data: {len(final)}')
        
        with open(out_file, 'w', encoding='utf-8') as f:
            for ann in final:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        print(f"Generated {len(final)} annotations and saved to {out_file}")

    else:
        for i, d in tqdm(enumerate(data)):
            #now we add also pre-trained text data to avoid forgetting
            text_c = d['text_c']
            given_string_text_c = " ".join(text_c)
            question = d['question']['text_only_num']
            ans = d['num_black']

            annotation_t = {
                "img_b64": None,
                "question_token": question,
                "conversations": [
                    {"from": "human", "value": [given_string_text_c]},
                    {"from": "gpt", "value": f"{ans:02d}"}
                ]
            }

            if (d['num_black'] <= text_boundary and i % 2 == 0):
                text_data.append(annotation_t)
            #-----this is for finetuning img num generation-----
            if (d['num_black'] > vision_boundary):
                 continue
            
            img_b64 = d['img_c']['image_b64']
            # img_bytes2 = base64.b64decode(d['img_c']['image_b64']) 
            # img2 = Image.open(BytesIO(img_bytes2))
            question = d['question']['multimodal_num']
            ans = d['num_black']
            ann = {
                "img_b64": img_b64,
                "question_token": question,
                "conversations": [
                    {"from": "human", "value": []},
                    {"from": "gpt", "value": f"{ans:02d}"}
                ]
            }
            annotations.append(ann)

        random.shuffle(annotations)
        random.shuffle(text_data)
        print(f'length of pre-trained text data: {len(text_data)}')
        print(f'length of finetuned vision data: {len(annotations)}')
        
        final = merge_blockwise(text_data, annotations, 2048)
        # final = annotations
        print(f'length of final finetuning data: {len(final)}')
        
        with open(out_file, 'w', encoding='utf-8') as f:
            for ann in final:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        print(f"Generated {len(final)} annotations and saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_type", type=str, required=True,
                        choices=["pretrain", "finetune"], help="pretrain for text-only pretraining, finetune for multimodal finetuning")
    parser.add_argument("--output_type", type=str, required=True,
                        choices=["tf", "num"], help="tf for true/false output, num for number output")
    parser.add_argument("--data_path", type=str, default='./training_data_formal.jsonl')
    parser.add_argument("--output_dir", type=str, default='./')
    parser.add_argument("--vision_boundary", type=int, default=49)
    parser.add_argument("--text_boundary", type=int, default=99)

    args = parser.parse_args()

    generate_json_annotations(
        process_type=args.process_type,
        output_type=args.output_type,
        vision_boundary=args.vision_boundary,
        text_boundary=args.text_boundary,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
