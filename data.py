"""
This script generates training and testing data for the textual and visual counting task. 
The generated data includes images of the Gomoku board, corresponding strings, and questions 
for both text-only and multimodal evaluation. The data is saved in JSONL format.
You can run the code in the following way: 
python data.py --training_data_point 8192 --model_type synthetic
"""
import os
import random
import base64
import json
import argparse

from tqdm import tqdm
from PIL import Image, ImageDraw
from io import BytesIO

# modality counting task alignment, ensuring it has the same difficulty across different modalities
CHAR_MAP = {
            'X': 'c',  # Black/X -> c
            'O': 'b',  # White/O -> b
            '.': 'a'   # Empty/. -> a
        }

def generate_str(list_str, target, count, seq_len, alphabet, rng):
    """"
    Generate two strings with the same count of target character, 
    and one string with a different count of target character. 
    Ensure all three strings are unique and not in the provided list_str set.
    """
    check = False
    while not check:
        other = [c for c in alphabet if c != target]
        chars = [rng.choice(other) for _ in range(seq_len)]
        for pos in rng.sample(range(seq_len), count):
            chars[pos] = target

        char_str1 = "".join(chars)
        if char_str1 not in list_str:
            list_str.add(char_str1)
            check = True

    check = False
    while not check:
        other = [c for c in alphabet if c != target]
        chars = [rng.choice(other) for _ in range(seq_len)]
        for pos in rng.sample(range(seq_len), count):
            chars[pos] = target

        char_str2 = "".join(chars)
        if char_str2 not in list_str:
            list_str.add(char_str2)
            check = True

    num_ie = random.randrange(seq_len)
    while (num_ie == count):
        num_ie = random.randrange(seq_len)

    other = [c for c in alphabet if c != target]
    chars = [rng.choice(other) for _ in range(seq_len)]
    for pos in rng.sample(range(seq_len), num_ie):
        chars[pos] = target

    char_str3 = "".join(chars)

    return char_str1, char_str2, char_str3, list_str

def generate_random_gomoku_state(size, list_boards, num_stones, distractor_range):
    """"
    Generate a random Gomoku board state with a specific number of black stones (num_stones) 
    and a random number of white stones within a certain range (distractor_range). 
    Ensure the generated board state is unique and not in the provided list_boards set.
    """
    check = False
    lower_bound = (num_stones - distractor_range) if ((num_stones - distractor_range) > 0) else 0
    upper_bound = num_stones + distractor_range
    board_str = None
    while not check:
        white = random.randint(lower_bound, upper_bound)
        black = num_stones
        white = min(white, size * size - black)
        num_moves = black + white
        slots = [(r, c) for r in range(size) for c in range(size)]
        available_slots = slots.copy()
        chosen_slots = []
        while (len(chosen_slots) < num_moves):
            r, c = random.choice(available_slots)
            chosen_slots.append((r,c))
            available_slots = [(r2, c2) for (r2, c2) in available_slots
                            if not (r2 == r and c2 == c)]
        
        board = [['.' for _ in range(size)] for _ in range(size)]
        
        for i, (r, c) in enumerate(chosen_slots):
            if i < black:
                board[r][c] = 'X'
            else:
                board[r][c] = 'O'

        board_str = ''.join(''.join(row) for row in board)
        #ensure the generated board state is unique
        if board_str not in list_boards:
            list_boards.add(board_str)
            check = True

    return num_moves, board, black, white, list_boards, board_str

def draw_board(board, stone_radius_px=5, cell_size_px=14):
    board_color='#DEB887'
    grid_line_width_px = 2

    rows = len(board)
    cols = len(board[0])
    content_height_px = cell_size_px * rows
    content_width_px = cell_size_px * cols
    
    img = Image.new('RGB', (content_width_px, content_height_px), board_color)
    draw = ImageDraw.Draw(img)

    for i in range(0, rows):
        offset = i * cell_size_px + cell_size_px // 2 - 1
        draw.line([(0, offset), (content_width_px, offset)],
                  fill='black', width=grid_line_width_px)
        
    for i in range(0, cols):
        offset = i * cell_size_px + cell_size_px // 2 - 1
        draw.line([(offset, 0), (offset, content_height_px)],
                  fill='black', width=grid_line_width_px)

    for r in range(rows):
        for c in range(cols):
            piece = board[r][c]
            if piece in ('X','O'):
                cx = c * cell_size_px + cell_size_px//2
                cy = r * cell_size_px + cell_size_px//2
                bbox = [ (cx-stone_radius_px, cy-stone_radius_px),
                         (cx+stone_radius_px, cy+stone_radius_px)]
                fill = 'black' if piece=='X' else 'white'
                draw.ellipse(bbox, fill=fill, outline=None)

    return img

if __name__ == '__main__':
    random.seed(42)

    parser = argparse.ArgumentParser(description="Generate training and testing data for visual counting task.")
    parser.add_argument("--training_data_point", type=int, default=8192, help="number of data points for each stone count")
    parser.add_argument("--model_type", type=str, default='synthetic', choices=['qwen3vl', 'synthetic'], help="type of model for which the data is being generated, which determines the board size and the range of stone counts")
    args = parser.parse_args()

    training_data_point = args.training_data_point
    model_type = args.model_type

    training_data = []
    testing_data = []
    testing_data_ood = []
    text_counting_alphabet = ['a', 'b', 'c']
    text_tgt = text_counting_alphabet[2]
    
    board_length = 19 if (model_type == 'synthetic') else 6
    max_stone = 120 if (model_type == 'synthetic') else 20
    testing_data_point = 1024 if (model_type == 'synthetic') else 100
    distractor_range = 30 if (model_type == 'synthetic') else 5
    vision_boundary = 49
    full_extrapolation_boundary = 99

    for num_stones in tqdm(range(0, max_stone + 1)):
        list_boards = set()
        list_str = set()
        if (num_stones <= full_extrapolation_boundary):
            for i in range(training_data_point):
                global_idx = num_stones * training_data_point + i
                num_moves, board, black, white, list_boards, board_str = generate_random_gomoku_state(size = board_length, list_boards=list_boards, num_stones = num_stones, distractor_range=distractor_range)
                img = draw_board(board)
                buf = BytesIO()
                img.save(buf, format="PNG") 
                img_bytes = buf.getvalue() 
                img_b64 = base64.b64encode(img_bytes).decode("ascii")
                
                text_c, text_eq, text_ie, list_str = generate_str(list_str, text_tgt, num_stones, board_length * board_length, text_counting_alphabet, random)

                curr_exp = {'global_idx': global_idx,
                            'num_black': num_stones,
                            'img_c': {"format": "PNG",
                                      "image_b64": img_b64,
                                     },
                            'str_c': board_str,
                            'text_c': text_c,
                            'text_eq': text_eq,
                            'text_ie': text_ie,
                            'question': {"text_only_tf": "Are the number of c letters in both input strings the same ?", 
                                         "text_only_num": "How many c letters are there in the given string ?", 
                                         "multimodal_tf": "Is the number of c letters in the given string the same as the number of black stones on the chessboard ?", 
                                         "multimodal_num": "How many black stones are there on the chessboard ?"}}

                training_data.append(curr_exp)

        for i in range(testing_data_point):
            global_idx_test = num_stones * testing_data_point + i
            num_moves, board, black, white, list_boards, board_str = generate_random_gomoku_state(size = board_length, list_boards=list_boards, num_stones = num_stones, distractor_range=distractor_range)
            if model_type == 'qwen3vl':
                img = draw_board(board, stone_radius_px=6, cell_size_px=16)
                img2 = draw_board(board, stone_radius_px=12, cell_size_px=32)
            else:
                img = draw_board(board, stone_radius_px=5, cell_size_px=14)
            buf = BytesIO()
            img.save(buf, format="PNG") 
            img_bytes = buf.getvalue() 
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            if model_type == 'qwen3vl':
                buf2 = BytesIO()
                img2.save(buf2, format="PNG") 
                img_bytes2 = buf2.getvalue() 
                img_b64_2 = base64.b64encode(img_bytes2).decode("ascii")
            
            text_c, text_eq, text_ie, list_str = generate_str(list_str, text_tgt, num_stones, board_length * board_length, text_counting_alphabet, random)

            curr_exp = {'global_idx': global_idx_test,
                        'num_black': num_stones,
                        'img_c': {"format": "PNG",
                                "image_b64": img_b64,
                                },
                        'str_c': board_str,
                        'text_c': text_c,
                        'text_eq': text_eq,
                        'text_ie': text_ie,
                        "question": {"text_only_tf": "Are the number of c letters in both input strings the same ?", 
                                     "text_only_num": "How many c letters are there in the given string ?", 
                                     "multimodal_tf": "Is the number of c letters in the given string the same as the number of black stones on the chessboard ?", 
                                     "multimodal_num": "How many black stones are there on the chessboard ?"}}
            
            if model_type == 'qwen3vl':
                text_board = ''.join([CHAR_MAP[c] for c in board_str])
                curr_exp['text_board'] = text_board
                curr_exp['img_c_2'] = {"format": "PNG",
                                      "image_b64": img_b64_2,
                                     }
                
                curr_exp['question'] = {"text_only_tf": "Compare the count of 'c' in List A vs List B. Are they equal?",

                    "text_only_num": "String length: 36. Count the occurrences of letter 'c'.",

                    "multimodal_tf": "Compare the count of 'c' in the text vs the black stones on the 6x6 board. Are they equal?",

                    "multimodal_num": "Analyze the 6x6 Go board. Count the black stones.",
                }
                curr_exp['system_prompt'] = "You are a precise counting engine. " + \
                                            "1. Verification: Before answering, output a compact data check (e.g., list of indices for text, or row-by-row counts for images). Avoid conversational filler. " + \
                                            "2. Format: Verification: <Dense Data> -> FINAL_ANSWER: <Result>."
            
            if num_stones > full_extrapolation_boundary:
                testing_data_ood.append(curr_exp)
            else:
                testing_data.append(curr_exp)
    
    # training_data_copy = training_data.copy()
    # random.shuffle(training_data_copy)
    if model_type == 'qwen3vl':
        testfile_qwen3vl_path = './testing_data_qwen3vl_6x6_0_to_20.jsonl'
        if os.path.exists(testfile_qwen3vl_path):
            os.remove(testfile_qwen3vl_path)
        with open(testfile_qwen3vl_path, 'w', encoding='utf-8') as f:
            for ann in testing_data:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")
    else:
        training_file_path = './training_data_formal.jsonl'
        if os.path.exists(training_file_path):
            os.remove(training_file_path)
        with open(training_file_path, 'w', encoding='utf-8') as f:
            for ann in training_data:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        testing_file_path = './testing_data_formal.jsonl'
        if os.path.exists(testing_file_path):
            os.remove(testing_file_path)
        with open(testing_file_path, 'w', encoding='utf-8') as f:
            for ann in testing_data:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        testing_ood_file_path = './testing_data_formal_ood.jsonl'
        if os.path.exists(testing_ood_file_path):
            os.remove(testing_ood_file_path)
        with open(testing_ood_file_path, 'w', encoding='utf-8') as f:
            for ann in testing_data_ood:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")