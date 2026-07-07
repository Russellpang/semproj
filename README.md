# visual_count

Code for my ETH MSc semester project on visual counting. You can access the original paper under the following link: https://arxiv.org/abs/2605.30170. The repository contains:

- a synthetic Go-board counting dataset generator,
- a small custom vision-language model for text-only pretraining and multimodal finetuning,
- evaluation scripts for the custom model and Qwen3VL,
- probing scripts that test whether hidden representations encode grid-level labels and black-stone counts,
- intervention and pruning-analysis utilities.

The core task is to count black stones on a board, or to compare that count against a text sequence with the same controlled counting structure.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `data.py` | Generates JSONL data for synthetic 19x19 custom-model experiments or 6x6 Qwen3VL experiments. |
| `generate_training_data_json.py` | Converts generated formal data into the chat-style JSONL format used by `model.py`. |
| `model.py` | Custom VLM implementation, processor, trainer, and training entry point. |
| `tokenizer.py` | Builds/loads the custom WordLevel tokenizer used by the custom VLM. |
| `mrope.py` | Multimodal RoPE utilities used by the custom VLM. |
| `test_model.py` | Runs generation evaluation for the custom VLM. |
| `test_acc.py` | Computes accuracy summaries from `test_model.py` JSONL logs. |
| `test_qwen3vl.py` | Runs Qwen3VL text-only or multimodal counting evaluation. |
| `train_hidden_num_qwen3vl.py` | Trains one Qwen3VL vision-feature probe using `get_image_features`. |
| `infer_hidden_num_qwen3vl_probe.py` | Runs inference with the saved Qwen3VL vision-feature probe. |
| `train_lm_layer_hidden_num_qwen3vl.py` | Trains one hidden-number probe per Qwen3VL LM hidden-state layer. |
| `infer_lm_layer_hidden_num_qwen3vl.py` | Evaluates all saved Qwen3VL LM-layer probes. |
| `hidden_number_probing_clf.py` | Trains a patch-level classifier on the custom VLM vision encoder. |
| `test_hidden_num.py` | Evaluates the custom VLM patch-level classifier by counting predicted black patches. |
| `causal_hidden_number_intervention.py` | Masks probe-positive image tokens and checks whether predicted counts drop by the masked amount. |
| `test_head_deactivation_text.py` / `test_head_deactivation_vision.py` | Tests custom-model attention-head deactivation effects. |
| `analyse_pruning_intensity.py` | Computes position-wise intensity statistics for 2x2 pruning vectors. |
| `compare_pruning_vectors.py` | Compares corresponding keys in two pruning-vector JSONL files. |

## Environment

The scripts assume Python 3.11.6 with CUDA for model training and Qwen3VL inference:

```bash
pip install torch torchvision transformers tokenizers pillow tqdm numpy scikit-learn joblib qwen-vl-utils
```

## Pipeline & Overall Structure

### Data Generation

#### Synthetic 19x19 Data For The Custom VLM

```bash
python data.py \
  --training_data_point 8192 \
  --model_type synthetic
```

This writes:

- `training_data_formal.jsonl`
- `testing_data_formal.jsonl`
- `testing_data_formal_ood.jsonl`

For `model_type=synthetic`, boards are 19x19 and counts range from 0 to 120. The default domain split used elsewhere is:

- in-domain: 0-49 black stones,
- visual extrapolation: 50-99 black stones,
- full extrapolation: 100-120 black stones.

#### 6x6 Qwen3VL Data

```bash
python data.py \
  --training_data_point 8192 \
  --model_type qwen3vl
```

This writes:

- `testing_data_qwen3vl_6x6_0_to_20.jsonl`

For Qwen3VL, boards are 6x6 and counts range from 0 to 20. Only testing data is generated and stored. Records include both image sizes:

- `img_c`: smaller rendered board with grid size 16 px * 16 px,
- `img_c_2`: larger rendered board aligned with Qwen3VL's default visual patch/merge setup: grid size 32 px * 32 px.

### Custom VLM Workflow

#### 1. Convert Formal Data To Training JSONL

Text-only pretraining data:

```bash
python generate_training_data_json.py \
  --process_type pretrain \
  --output_type num \
  --data_path ./training_data_formal.jsonl
```

Multimodal finetuning data:

```bash
python generate_training_data_json.py \
  --process_type finetune \
  --output_type num \
  --data_path ./training_data_formal.jsonl
```

Use `--output_type tf` for true/false comparison experiments. Output filenames encode the process, output type, vision boundary, and text boundary, for example:

- `training_19x19_text_pretrain_output_num_vision_49_text_99.jsonl`
- `training_19x19_multimodal_finetune_output_num_vision_49_text_99.jsonl`

#### 2. Train Text-Only Pretraining Model

```bash
python model.py \
  --data_path ./training_19x19_text_pretrain_output_num_vision_49_text_99.jsonl \
  --mode train \
  --output_type num
```

The tokenizer is created automatically at `./tokenizer_formal.json` if it does not already exist.

#### 3. Finetune Multimodal Model

```bash
python model.py \
  --data_path ./training_19x19_multimodal_finetune_output_num_vision_49_text_99.jsonl \
  --model_path ./custom-model-pretrained-num-vision-50-text-100 \
  --mode finetune \
  --output_type num
```

For finetuning, `--model_path` is required and must match the requested `--output_type`.

#### 4. Evaluate The Custom Model

```bash
python test_model.py \
  --mode multimodal \
  --output_type num \
  --model_type finetune \
  --model_path ./custom-model-finetune-num-vision-50-text-100
```

Text-only evaluation:

```bash
python test_model.py \
  --mode text \
  --output_type num \
  --model_type finetune \
  --model_path ./custom-model-finetune-num-vision-50-text-100
```

Accuracy helper:

```bash
python test_acc.py \
  --log_file ./answer-output-finetune-multimodal-num-vision-50-text-100.log \
  --output_type num
```

`test_acc.py` currently builds the summary but does not print it because the relevant print loops are commented out near the end of the file. Uncomment the desired block to print per-domain accuracy, per-count accuracy, average error, or generated-number frequencies.

### Qwen3VL Workflow

#### Run Qwen3VL Evaluation

```bash
python test_qwen3vl.py \
  --output_type num \
  --input_type multimodal \
  --model_path Qwen/Qwen3-VL-32B-Instruct
```

Text-only Qwen3VL evaluation:

```bash
python test_qwen3vl.py \
  --output_type num \
  --input_type text_only \
  --model_path Qwen/Qwen3-VL-32B-Instruct
```

The Qwen3VL prompt asks the model to emit a dense verification trace followed by `FINAL_ANSWER: <Result>`. If no final answer is found, `test_qwen3vl.py` appends `FINAL_ANSWER: -1.` to the log.

### Qwen3VL Probing

The Qwen3VL probe scripts intentionally split the 6x6 JSONL by line index:

- training probes use lines where `idx >= 100 and idx % 10 == 9`, 
- inference probes use the remaining lines.

This keeps a balanced held-out subset for testing.

#### Vision Feature Probe

Train a probe on Qwen3VL image features:

```bash
python train_hidden_num_qwen3vl.py --model_path Qwen/Qwen3-VL-32B-Instruct 
```

This saves:

- `./qwen3vl/probe_output/vision_get_image_features_probe.joblib`

Run inference:

```bash
python infer_hidden_num_qwen3vl_probe.py \
  --model_path Qwen/Qwen3-VL-32B-Instruct \
  --probe_path ./qwen3vl/probe_output/vision_get_image_features_probe.joblib \
  --out_file ./qwen3vl/probe_output/rest_probe_predictions.jsonl
```

#### LM Hidden-State Layer Probes

Train one probe per Qwen3VL language-model hidden-state layer:

```bash
python train_lm_layer_hidden_num_qwen3vl.py \
  --model_path Qwen/Qwen3-VL-32B-Instruct \
  --out_dir ./qwen3vl/lm_layer_probe_output
```

Each saved file is named:

```text
lm_hidden_state_XX_probe.joblib
```

Run per-layer inference:

```bash
python infer_lm_layer_hidden_num_qwen3vl.py \
  --model_path Qwen/Qwen3-VL-32B-Instruct \
  --probe_dir ./qwen3vl/lm_layer_probe_output \
  --out_file ./qwen3vl/lm_layer_probe_output/rest_lm_layer_probe_predictions.jsonl
```

The output JSONL contains the ground-truth `num_black` and each layer probe's predicted black-stone count.

### Custom VLM Probing And Interventions

#### Train Patch-Level Classifier

```bash
python hidden_number_probing_clf.py \
  --model_path ./custom-model-finetune-num-vision-50-text-100 \
  --output_dir ./prediction_head
```

This saves:

- `./prediction_head/prediction_head_vit.joblib`

#### Evaluate Patch-Level Classifier

```bash
python test_hidden_num.py \
  --model_path ./custom-model-finetune-num-vision-50-text-100 \
  --head_path ./prediction_head/prediction_head_vit.joblib
```

#### Causal Hidden-Number Intervention

```bash
python causal_hidden_number_intervention.py \
  --model_path ./custom-model-finetune-num-vision-50-text-100 \
  --head_path ./prediction_head/prediction_head_vit.joblib \
  --data_path ./testing_data_formal.jsonl \
  --output_path ./causal_hidden_number_accuracy.json \
  --max_mask_count 5
```

For each valid in-domain example, this script:

1. extracts vision-token embeddings,
2. predicts black-stone token indices using the patch classifier,
3. masks `k` predicted black-stone tokens,
4. checks whether the generated count becomes `ground_truth - k`.

### Pruning Utilities

Test the effect of deactivating individual custom-model attention heads for text inputs:

```bash
python test_head_deactivation_text.py \
  --model_path ./custom-model-finetune-num-vision-50-text-100 \
  --stride 8 \
  --batch_size 1
```

Test the effect of deactivating individual custom-model attention heads for vision inputs:

```bash
python test_head_deactivation_vision.py \
  --model_path ./custom-model-finetune-num-vision-50-text-100 \
  --stride 8 \
  --batch_size 1
```

Both scripts iterate over decoder layers and heads, disable one head at a time during generation, and report the resulting counting accuracy. Use `--stride` to subsample the test set for faster sweeps.

Analyze position-wise intensity in nested 2x2 pruning vectors:

```bash
python analyse_pruning_intensity.py \
  --path pruning_vectors_text.jsonl \
  --key keep

python analyse_pruning_intensity.py \
  --path pruning_vectors_vision.jsonl \
  --key keep
```

Compare corresponding JSONL records:

```bash
python compare_pruning_vectors.py \
  --vision pruning_vectors_vision.jsonl \
  --text pruning_vectors_text.jsonl \
  --key keep
```
