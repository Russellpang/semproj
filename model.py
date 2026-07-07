"""
This script implements a custom vision-language model. It includes training and fine-tuning 
capabilities using Hugging Face's Trainer API, with support for saving checkpoints every N epochs.
You can run the code in the following way: 
For pretraining: python model.py --data_path './training_19x19_text_pretrain_output_num_vision_49_text_99.jsonl' --mode train --output_type num 
For finetuning: python model.py --data_path './training_19x19_multimodal_finetune_output_num_vision_49_text_99.jsonl' --model_path 'custom-model-pretrained-num-vision-50-text-100' --mode 'finetune' --output_type 'num'
"""
import base64
import torch
import shutil
import json
import math
import os
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from io import BytesIO 
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedModel, PretrainedConfig, ProcessorMixin, BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.processing_auto import AutoProcessor
from transformers import TrainerCallback

from tokenizer import create_tokenizer
from mrope import MultimodalRoPE, create_3d_positions_for_images

device = 'cuda'

def apply_disable_indices(mask, disable_indices):
    if disable_indices is None or len(disable_indices) == 0:
        return mask

    for row_idx, row_indices in enumerate(disable_indices):
        if len(row_indices) == 0:
            continue
        idx_tensor = torch.as_tensor(row_indices, device=mask.device, dtype=torch.long)
        mask[row_idx, idx_tensor] = False
    return mask

class LrGroupTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            vision_params = []
            # image_proj_params = []
            decoder_params = []
            other_params = []

            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("vision_encoder"):
                    vision_params.append(p)
                # elif n.startswith("decoder.image_projection"):
                #     image_proj_params.append(p)
                elif n.startswith("decoder"):
                    decoder_params.append(p)
                else:
                    other_params.append(p)

            param_groups = [
                {"params": vision_params, "lr": 1e-3},
                # {"params": image_proj_params, "lr": 1e-3},
                {"params": decoder_params, "lr": 1e-5},
            ]
            if other_params:
                param_groups.append({"params": other_params, "lr": 1e-3})

            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            sampler=SequentialSampler(self.train_dataset),
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=getattr(self.args, "dataloader_persistent_workers", False),
        )


class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every_n_epochs, save_dir):
        self.n = save_every_n_epochs
        self.save_dir = save_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return control

        epoch_int = int(round(state.epoch))
        if epoch_int > 0 and (epoch_int % self.n == 0):
            out = os.path.join(self.save_dir, f"epoch-{epoch_int}")
            os.makedirs(out, exist_ok=True)

            model = kwargs["model"]
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(out)  # saves weights + config if config exists
            else:
                torch.save(model.state_dict(), os.path.join(out, "pytorch_model.bin"))

            tok = kwargs.get("tokenizer", None) or kwargs.get("processing_class", None)
            if tok is not None and hasattr(tok, "save_pretrained"):
                tok.save_pretrained(out)

        return control

class VLMConfig(PretrainedConfig):
    model_type = "custom_vlm"
    
    def __init__(
        self,
        vocab_size=64,
        n_embd=32,
        image_embed_dim=32,
        n_layer=2,
        img_size=266,
        patch_size=14,
        n_head=2,
        num_blks=2,
        dropout=0,
        stop_id=None,
        image_start_id=None,
        image_end_id=None,
        # M-RoPE specific parameters
        use_mrope=True,
        mrope_type="standard",
        mrope_section=[4, 6, 6],  # [temporal, height, width] dimensions
        rope_base=10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_embd = n_embd
        self.image_embed_dim = image_embed_dim
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_head = n_head
        self.num_blks = num_blks
        self.dropout = dropout
        self.stop_id = stop_id
        self.image_start_id = image_start_id
        self.image_end_id = image_end_id
        # M-RoPE parameters
        self.use_mrope = use_mrope
        self.mrope_type = mrope_type
        self.mrope_section = mrope_section
        self.rope_base = rope_base


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=266, patch_size=14, is_rgb = True, embed_dim=32):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.channel = 3 if is_rgb else 1
        self.proj = nn.Conv2d(in_channels=self.channel, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class Head(nn.Module):
    def __init__(self, embed_dim, head_size, dropout=0, is_decoder=False, use_mrope=False):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.is_decoder = is_decoder
        self.use_mrope = use_mrope
        self.last_attn_weights = None

    def forward(self, x, key_padding_mask=None, rope_module=None, positions_3d=None):
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        if self.use_mrope and rope_module is not None and positions_3d is not None:
            # Reshape for M-RoPE: (batch, seq_len, n_heads=1, head_dim)
            q = q.unsqueeze(2)
            k = k.unsqueeze(2)
            q, k = rope_module(q, k, positions_3d)
            q = q.squeeze(2)
            k = k.squeeze(2)
        
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        if self.is_decoder:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            wei = wei.masked_fill(mask == 0, float('-inf'))

        if key_padding_mask is not None:
            kp = key_padding_mask.to(torch.bool).unsqueeze(1)
            wei = wei.masked_fill(~kp, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        if not self.training:
            self.last_attn_weights = wei.detach().cpu()
        out = wei @ v
        
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, is_decoder=False, use_mrope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            Head(embed_dim, self.head_dim, dropout, is_decoder, use_mrope)
            for _ in range(self.num_heads)
        ])
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(self, x, key_padding_mask=None, rope_module=None, positions_3d=None, head_mask=None):
        if head_mask is None:
            hm = None
        else:
            hm = torch.as_tensor(head_mask, device=x.device, dtype=x.dtype)
            hm = hm.view(1, self.num_heads).expand(x.size(0), self.num_heads)
        
        head_outputs = []
        head_weights = []
        for i, h in enumerate(self.heads):
            out_i = h(x, key_padding_mask, rope_module, positions_3d)
            if hm is not None:
                out_i = out_i * hm[:, i].view(x.size(0), 1, 1)
            head_outputs.append(out_i)

            if not self.training and h.last_attn_weights is not None:
                head_weights.append(h.last_attn_weights)

        if not self.training and len(head_weights) == len(self.heads):
            self.last_attn_weights = torch.stack(head_weights, dim=1)
        
        out = torch.cat(head_outputs, dim=-1)
        
        out = self.proj(out)
        
        out = self.dropout(out)
        
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super().__init__()
        
        layers = [
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        ]
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, is_decoder=False, use_mrope=False):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout, is_decoder, use_mrope)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, dropout)

    def forward(self, x, key_padding_mask=None, rope_module=None, positions_3d=None, head_mask=None):
        original_x = x
        
        x = self.ln1(x)
        attn_output = self.attn(x, key_padding_mask, rope_module, positions_3d, head_mask)
        x = original_x + attn_output
        
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        
        return x


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_blks, drop_out, 
                 use_mrope=False, mrope_module=None):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, is_rgb=True, embed_dim=embed_dim)
        
        self.img_size = img_size
        self.patch_size = patch_size
        # num_patches = (img_size // patch_size) ** 2
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        
        # Use M-RoPE or traditional positional embedding
        self.use_mrope = use_mrope
        self.mrope_module = mrope_module
        
        # if not use_mrope:
        #     # Traditional positional embedding
        #     self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, drop_out, is_decoder=False, use_mrope=use_mrope) 
            for _ in range(num_blks)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, X, return_grid_dims=False, disable_indices=None):
        B = X.shape[0]
        x = self.patch_embedding(X)
        N = x.shape[1]
        
        positions_3d = None
        if self.use_mrope and self.mrope_module is not None:
            positions_3d = create_3d_positions_for_images(
                batch_size=B,
                num_patches_h=self.num_patches_h,
                num_patches_w=self.num_patches_w,
                num_frames=1,
                device=x.device,
            )

        key_padding_mask = None
        if disable_indices is not None and len(disable_indices) > 0:
            key_padding_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            key_padding_mask = apply_disable_indices(key_padding_mask, disable_indices)

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, rope_module=self.mrope_module if positions_3d is not None else None, positions_3d=positions_3d)

        x = self.layer_norm(x)
        
        if return_grid_dims:
            return x, (self.num_patches_h, self.num_patches_w)
        return x


class DecoderLanguageModel(nn.Module):

    def __init__(self, embed_dim, image_embed_dim, vocab_size, num_heads, num_blks, multimodal=True, stop_id=None,
                 use_mrope=False, mrope_module=None, image_start_id=None, image_end_id=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.multimodal = multimodal
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_blks = num_blks
        self.stop_id = stop_id
        self.use_mrope = use_mrope
        self.mrope_module = mrope_module
        self.image_start_id = image_start_id
        self.image_end_id = image_end_id

        self._cache_after_block0 = None
        self._cache_image_len = None
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # if not use_mrope:
        #     # Traditional positional embedding for text
        #     self.pos_embedding = nn.Embedding(512, embed_dim)
        
        #self.dropout = nn.Dropout(0.1)
        
        if self.multimodal:
            self.image_projection = nn.Linear(image_embed_dim, embed_dim)
        
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, 0, is_decoder=True, use_mrope=use_mrope) 
            for _ in range(num_blks)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def _get_image_boundary_embeds(self, batch_size, device, dtype):
        start_ids = torch.full((batch_size, 1), self.image_start_id, device=device, dtype=torch.long)
        end_ids = torch.full((batch_size, 1), self.image_end_id, device=device, dtype=torch.long)
        start_embeds = self.token_embedding(start_ids)
        end_embeds = self.token_embedding(end_ids)
        return start_embeds, end_embeds
    
    def _build_positions(self, image_len, text_len, batch_size, device, include_image_tokens=False):
        extra_tokens = 2 if include_image_tokens else 0
        text_positions = None
        if text_len > 0:
            text_positions = torch.zeros(batch_size, text_len, 3, device=device)
            steps = torch.arange(text_len, device=device).unsqueeze(0)
            text_positions[:, :, 0] = steps + image_len + extra_tokens
            text_positions[:, :, 1] = text_positions[:, :, 0]
            text_positions[:, :, 2] = text_positions[:, :, 0]

        if image_len == 0:
            return text_positions if text_positions is not None else torch.zeros(batch_size, 0, 3, device=device)

        img_h = int(math.sqrt(image_len))
        img_w = image_len // img_h

        img_positions = create_3d_positions_for_images(
            batch_size=batch_size,
            num_patches_h=img_h,
            num_patches_w=img_w,
            device=device,
        )

        if include_image_tokens:
            start_pos = torch.zeros(batch_size, 1, 3, device=device)
            end_pos = torch.full((batch_size, 1, 3), image_len + 1, device=device)
            parts = [start_pos, img_positions, end_pos]
            if text_positions is not None:
                parts.append(text_positions)
            return torch.cat(parts, dim=1)

        if text_positions is None:
            return img_positions
        return torch.cat([img_positions, text_positions], dim=1)

    def forward(
        self,
        input_ids,
        image_embeds=None,
        targets=None,
        key_padding_mask=None,
        positions_3d=None,
        disable_indices=None,
        labels=None,
        attention_mask=None,
        head_mask=None,
    ):
        B, T = input_ids.shape
        if key_padding_mask is None and attention_mask is not None:
            key_padding_mask = attention_mask
        if targets is None and labels is not None:
            targets = labels
        token_embeds = self.token_embedding(input_ids)

        x = token_embeds
        image_len = 0
        image_patch_len = 0
        combined_mask = key_padding_mask

        if self.multimodal and image_embeds is not None:
            image_embeds = self.image_projection(image_embeds)
            image_patch_len = image_embeds.shape[1]
            image_len = image_patch_len + 2
            start_embeds, end_embeds = self._get_image_boundary_embeds(B, token_embeds.device, token_embeds.dtype)
            x = torch.cat([start_embeds, image_embeds, end_embeds, token_embeds], dim=1)

            text_mask = key_padding_mask
            if text_mask is None:
                text_mask = torch.ones(B, T, dtype=torch.bool, device=input_ids.device)
            else:
                text_mask = text_mask.to(torch.bool)

            img_mask = torch.ones(B, image_patch_len, dtype=torch.bool, device=input_ids.device)

            if disable_indices is not None and len(disable_indices) > 0:
                img_mask = apply_disable_indices(img_mask, disable_indices)

            start_mask = torch.ones(B, 1, dtype=torch.bool, device=input_ids.device)
            end_mask = torch.ones(B, 1, dtype=torch.bool, device=input_ids.device)
            combined_mask = torch.cat([start_mask, img_mask, end_mask, text_mask], dim=1)

        if head_mask is not None:
            head_mask = torch.as_tensor(head_mask, device=input_ids.device)

        if self.use_mrope and self.mrope_module is not None:
            if positions_3d is None:
                positions_3d = self._build_positions(image_patch_len, T, B, x.device, include_image_tokens=image_len > 0)

            for i, block in enumerate(self.blocks):
                layer_mask = None if head_mask is None else head_mask[i]
                x = block(x, combined_mask, self.mrope_module, positions_3d, layer_mask)
                if i == 0:
                    self._cache_after_block0 = x
                    self._cache_image_len = image_len

        x = self.layer_norm(x)

        if image_len > 0:
            x = x[:, image_len:, :]

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

    def generate(self, input_ids, image_embeds, attention_mask, max_new_tokens, disable_indices=None, headmask=None):
        generated_ids = input_ids.clone()
        B, _ = input_ids.shape
        device = generated_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # positions_3d = None
        # image_len = 0
        # if self.multimodal and image_embeds is not None:
        #    img_emb = self.image_projection(image_embeds)
        #    image_len = image_embeds.shape[1]

        # if self.use_mrope and self.mrope_module is not None:
        #    positions_3d = self._build_positions(image_len, T_initial, B, device, include_image_tokens=image_len > 0)

        # boundary_tokens = 2 if image_len > 0 else 0
        # position_cursor = image_len + boundary_tokens + T_initial

        for _ in range(max_new_tokens):
            out = self.forward(
                generated_ids,
                image_embeds=image_embeds,
                key_padding_mask=attention_mask,
                positions_3d=None,
                disable_indices=disable_indices,
                head_mask=headmask
            )
            logits = out.logits

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            new_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

            # if positions_3d is not None:
            #     new_pos = torch.zeros(B, 1, 3, device=device)
            #     new_pos[:, :, 0] = position_cursor
            #     new_pos[:, :, 1] = position_cursor
            #     new_pos[:, :, 2] = position_cursor
            #     positions_3d = torch.cat([positions_3d, new_pos], dim=1)
            #     position_cursor += 1

            if self.stop_id is not None and (next_token == self.stop_id).all():
                break

        return generated_ids
    
class VisionLanguageModel(PreTrainedModel):
    config_class = VLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Calculate per-head dimension (must be even for RoPE)
        head_dim = config.image_embed_dim // config.n_head
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even to use Rotary Embeddings")
        half_head_dim = head_dim // 2
        
        # Create M-RoPE module if enabled
        self.mrope_module = None
        if config.use_mrope:
            scaled_section = None
            if config.mrope_section:
                scaled_section = self._scale_mrope_sections(config.mrope_section, half_head_dim)
            
            if config.mrope_type == "standard":
                self.mrope_module = MultimodalRoPE(
                    head_dim=head_dim,  # Use head_dim instead of embed_dim
                    mrope_section=scaled_section,
                    base=config.rope_base
                )
        
        mrope_module = self.mrope_module if config.use_mrope else None

        self.vision_encoder = ViT(
            config.img_size,
            config.patch_size,
            config.image_embed_dim,
            config.n_head,
            config.num_blks,
            config.dropout,
            use_mrope=config.use_mrope,
            mrope_module=mrope_module,
        )

        self.decoder = DecoderLanguageModel(
            config.n_embd,
            config.image_embed_dim,
            config.vocab_size,
            config.n_head,
            config.n_layer,
            True,
            config.stop_id,
            use_mrope=config.use_mrope,
            mrope_module=mrope_module,
            image_start_id=config.image_start_id,
            image_end_id=config.image_end_id,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        return_dict=True,
        disable_indices=None,
        head_mask=None, 
        **kwargs
    ):
        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.vision_encoder(pixel_values, disable_indices=disable_indices)
        out = self.decoder(
            input_ids,
            image_embeds=image_embeds,
            labels=labels,
            attention_mask=attention_mask,
            disable_indices=disable_indices,
            head_mask=head_mask
        )

        return CausalLMOutputWithPast(
            loss=out.loss,
            logits=out.logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

    def generate(self, input_ids, images=None, attention_mask=None, max_new_tokens=10, disable_indices=None, headmask=None, input_image_embeds=None):
        image_embeds = None
        if input_image_embeds is not None:
            image_embeds = input_image_embeds
        elif images is not None:
            image_embeds = self.vision_encoder(images, disable_indices=disable_indices)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated_tokens = self.decoder.generate(input_ids, image_embeds, attention_mask, max_new_tokens, disable_indices, headmask)
        return generated_tokens

    def _scale_mrope_sections(self, section_values, target_sum):
        import math as _math

        total = sum(section_values)
        if total <= 0:
            raise ValueError("mrope_section entries must sum to a positive value")

        raw = [value / total * target_sum for value in section_values]
        scaled = [int(_math.floor(val)) for val in raw]
        fractional = [val - _math.floor(val) for val in raw]

        remaining = target_sum - sum(scaled)
        if remaining > 0:
            order = sorted(range(len(scaled)), key=lambda i: fractional[i], reverse=True)
            for i in range(remaining):
                scaled[order[i % len(order)]] += 1
        elif remaining < 0:
            order = sorted(range(len(scaled)), key=lambda i: fractional[i])
            idx = 0
            while remaining < 0 and idx < len(order):
                target = order[idx]
                if scaled[target] > 0:
                    scaled[target] -= 1
                    remaining += 1
                else:
                    idx += 1
            if remaining < 0:
                raise ValueError("Cannot distribute mrope_section to match head dimension")

        for i, value in enumerate(scaled):
            if value <= 0:
                donor = max(range(len(scaled)), key=lambda j: scaled[j])
                if scaled[donor] <= 1:
                    raise ValueError("head_dim too small for given mrope_section")
                scaled[donor] -= 1
                scaled[i] = 1

        if sum(scaled) != target_sum:
            raise ValueError("Failed to scale mrope_section to desired head dimension")

        return scaled


class SimpleImageProcessor:
    def __init__(self):
        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
        ])

    def __call__(self, images, return_tensors="pt"):
        batch = []
        for im in images:
            if isinstance(im, torch.Tensor):
                batch.append(im)
            else:
                batch.append(self.tf(im))
        pixel_values = torch.stack(batch, dim=0)  # [B,3,H,W]
        return {"pixel_values": pixel_values}


class CustomVLMProcessor(ProcessorMixin):
    attributes = ["tokenizer"]#"image_processor", "tokenizer"]
    #image_processor_class = SimpleImageProcessor
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor, tokenizer):
        super().__init__(tokenizer = tokenizer)#(image_processor = image_processor, tokenizer = tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.user_start = "<|user_start|>"
        self.user_end = "<|user_end|>"
        self.assistant_start = "<|assistant_start|>"
        self.assistant_end = "<|assistant_end|>"
        self.image_start = "<|image_start|>"
        self.image_end = "<|image_end|>"
        self.image_token = "<|image|>"
        self.question_start = "<|question_start|>"
        self.question_end = "<|question_end|>"
        self.str_start = "<|str_start|>"
        self.str_end = "<|str_end|>"

    def apply_chat_template(self, messages, question, tokenize=False, add_generation_prompt=False, inference = False):
        """
        messages: list of dicts like your dataset builds:
          [{"role":"user","content":[{"type":"image","image":...},{"type":"text","text":"..."}]},
           {"role":"assistant","content":"..."}]
        Returns a single string with our markers.
        """
        parts = []
        parts.append(self.tokenizer.bos_token)
        for m in messages:
            role = m["from"]   # "human" or "gpt"
            text = m["value"]
            if role == "human":
                parts.append(self.user_start)
                for s in text:
                    parts.append(self.str_start)
                    parts.append(s.strip())
                    parts.append(self.str_end)
                #parts.append(text.strip())
                parts.append(self.question_start)
                parts.append(question)
                parts.append(self.question_end)
                parts.append(self.user_end)
            elif role == "gpt":
                parts.append(self.assistant_start)
                parts.append(text.strip())
                parts.append(self.assistant_end)

        if not inference:
            parts.append(self.tokenizer.eos_token)
        else:
            parts.append(self.assistant_start)

        s = " ".join(parts).strip()

        if tokenize:
            return self.tokenizer(s, add_special_tokens=False)["input_ids"]
        
        return s

    def __call__(self, text=None, images=None, return_tensors="pt", padding=True, **kwargs):
        enc = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=False,
            add_special_tokens=False,
        )
        if images is None:
            return BatchEncoding(data={**enc})
        img = self.image_processor(images, return_tensors=return_tensors)
        return BatchEncoding(data={**enc, **img})

class CustomDataset(Dataset):
    def __init__(self, data_path, modality_filter=None):
        def get_data(data_path = data_path):
            with open(data_path, 'r') as f:
                data = [json.loads(line) for line in f]
            return data
        
        self.data = get_data()
        self.modalities = []
        self.indices = []

        allowed = None
        if modality_filter is not None:
            if isinstance(modality_filter, (list, tuple, set)):
                allowed = set(modality_filter)
            else:
                allowed = {modality_filter}

        for i, item in enumerate(self.data):
            img = item.get("img_b64")
            modality = "image_text" if img else "text_only"
            if allowed is not None and modality not in allowed:
                continue
            self.indices.append(i)
            self.modalities.append(modality)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item = self.data[self.indices[idx]]
        img_b64 = item.get('img_b64')
        question_token = item.get("question_token", "")
        conversation = item['conversations']
        return idx, img_b64, question_token, conversation, self.modalities[idx]


def create_collate_fn(processor):
    def train_collate_fn(batch):
        idx, img_b64, questions, convs, modalities = zip(*batch)
        
        batch_modality = modalities[0]
        if any(m != batch_modality for m in modalities):
            raise ValueError("Mixed modalities in a single batch; check the batch sampler.")
        
        texts = [
            processor.apply_chat_template(msg, q, tokenize=False, add_generation_prompt=False, inference=False)
            for msg, q in zip(convs, questions)
        ]
        
        if batch_modality == "image_text":
            img_bytes2 = [base64.b64decode(i) for i in img_b64]
            original_pixel_values = [(Image.open(BytesIO(p))).convert("RGB") for p in img_bytes2]
            model_inputs = processor(
                text=texts, 
                images=original_pixel_values, 
                return_tensors="pt", 
                padding=True
            )
        else:
            model_inputs = processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = input_ids.clone()
        #batch_size = labels.shape[0]
        assistant_id_start = processor.tokenizer.convert_tokens_to_ids(processor.assistant_start)
        #assistant_id_end = processor.tokenizer.convert_tokens_to_ids(processor.assistant_end)
        eos   = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.eos_token)
        pad_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)

        T = input_ids.size(1)
        assistant_start_mask = (input_ids == assistant_id_start)
        eos_mask = (input_ids == eos)

        assistant_start_pos = assistant_start_mask.float().argmax(dim=1)  # (B,)
        eos_pos = eos_mask.float().argmax(dim=1)  # (B,)

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1,T)

        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        labels[pos <= assistant_start_pos.unsqueeze(1)] = -100
        labels[pos >  eos_pos.unsqueeze(1)] = -100

        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if batch_modality == "image_text":
            batch_out["pixel_values"] = model_inputs["pixel_values"]
        return batch_out
    
    return train_collate_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument('--data_path', type=str, required=True, help="Path to the data for pretraining or finetuning")
    parser.add_argument('--model_path', type=str, help="Path to the pre-trained model for finetuning")
    parser.add_argument("--mode", choices=["train", "finetune"], default="train")
    parser.add_argument("--output_type", choices=["num", "tf"], default="num")
    parser.add_argument("--vision_boundary", type=int, default=49)
    parser.add_argument("--text_boundary", type=int, default=99)
    parser.add_argument('--emb_dim', type=int, default=32, help="Embedding dimension for both vision and text")
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--n_layer', type=int, default=2, help="Number of transformer layers in encoder")
    parser.add_argument('--num_blks', type=int, default=2, help="Number of transformer blocks in decoder")
    parser.add_argument('--num_epoch', type=int, default=20, help="Number of training/finetuning epochs")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training/finetuning")

    args = parser.parse_args()

    data_path = args.data_path
    model_path = args.model_path
    mode = args.mode
    output_type = args.output_type
    vision_num = args.vision_boundary
    text_num = args.text_boundary
    emb_dim = args.emb_dim
    num_heads = args.num_heads
    n_layer = args.n_layer
    num_blks = args.num_blks
    num_epochs = args.num_epoch
    batch_size = args.batch_size

    if mode == "finetune" and not model_path:
        raise ValueError("model_path is required for finetuning mode")
    elif mode == "finetune" and (output_type not in data_path or output_type not in model_path):
        raise ValueError("Output_type with model_type or data_type unmatched!")
    
    if mode not in data_path or output_type not in data_path:
        raise ValueError("data_path must match the mode and output_type")        

    tokenizer = create_tokenizer('./tokenizer_formal.json', './training_data_formal.jsonl')
    vocab_size = len(tokenizer)
    processor = CustomVLMProcessor(SimpleImageProcessor(), tokenizer)
    image_start_id = tokenizer.convert_tokens_to_ids("<|image_start|>")
    image_end_id = tokenizer.convert_tokens_to_ids("<|image_end|>")
    
    if (mode == 'train'):
        config = VLMConfig(
            vocab_size=vocab_size,
            n_embd=emb_dim,
            image_embed_dim=emb_dim,
            n_layer=n_layer,
            n_head=num_heads,
            num_blks=num_blks,
            stop_id=tokenizer.eos_token_id,
            image_start_id=image_start_id,
            image_end_id=image_end_id,
            
            use_mrope=True,
            mrope_type="standard",
            # Note: mrope_section will be automatically scaled to head_dim
            mrope_section=[4,6,6],
            spatial_reset=True,
            rope_base=10000.0
        )

        model = VisionLanguageModel(config)
        output_dir= f"./vlm_pretrained_{output_type}_vision_{vision_num + 1}_text_{text_num + 1}"
        logging_dir= f"./training_logs_vlm_pretrained_{output_type}_vision_{vision_num + 1}_text_{text_num + 1}"
        callback_dir = f"./snapshots_{output_type}_vision_{vision_num + 1}_text_{text_num + 1}"
        data_path = data_path
    else:
        model = VisionLanguageModel.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        output_dir=f"./vlm_finetune_{output_type}_vision_{vision_num + 1}_text_{text_num + 1}"
        logging_dir=f"./training_logs_vlm_finetune_{output_type}_vision_{vision_num + 1}_text_{text_num + 1}"
        callback_dir = f"./snapshots_{output_type}_vision_{vision_num + 1}_text_{text_num + 1}"
        data_path = data_path

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    if os.path.exists(logging_dir):
        shutil.rmtree(logging_dir)
        
    if os.path.exists(callback_dir):
        shutil.rmtree(callback_dir)
    
    train_dataset = CustomDataset(data_path=data_path)
    data_collator = create_collate_fn(processor=processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-3,
        weight_decay=0.01,
        logging_steps=10,
        logging_dir=logging_dir,
        save_strategy="no",
        #save_steps=500,
        eval_strategy="no",
        warmup_steps=50,
        #lr_scheduler_type="cosine",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        optim="adamw_torch",
    )

    if (mode == 'train'):
        trainer = Trainer(
            model=model.decoder,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=processor.tokenizer,
            callbacks=[SaveEveryNEpochsCallback(5, callback_dir)],
        )
        model_save_path = f"./custom-model-pretrained-{output_type}-vision-{vision_num + 1}-text-{text_num + 1}"
    else:
        trainer = LrGroupTrainer(model=model, 
                             args=training_args, 
                             train_dataset=train_dataset, 
                             data_collator=data_collator, 
                             tokenizer=processor.tokenizer,
                             callbacks=[SaveEveryNEpochsCallback(5, callback_dir)])
        
        model_save_path = f"./custom-model-finetune-{output_type}-vision-{vision_num + 1}-text-{text_num + 1}"
    
    trainer.train()
    model.save_pretrained(model_save_path)
