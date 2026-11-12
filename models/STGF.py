import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from models.GPT2_arch import AccustumGPT2Model


class NumericEncoder(nn.Module):
    """CALF's numerical encoder: one token per variable.

    Linear(L -> d) followed by one Transformer encoder layer across variables
    (Eq. 3).  Changing the input length only changes the input projection.
    """

    def __init__(self, input_dim, hidden_dim=768, num_heads=12):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.linear(x)
        return self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)


class MemoryCrossAttention(nn.Module):
    """Variable tokens (queries) read a frozen PCA memory (keys/values), Eq. 4."""

    def __init__(self, width, rank=64, heads=4):
        super().__init__()
        self.rank, self.heads = rank, heads
        self.numeric_norm = nn.LayerNorm(width)
        self.text_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, rank, bias=False)
        self.key = nn.Linear(width, rank, bias=False)
        self.value = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(rank, width, bias=False)

    def forward(self, numeric, memory):
        B, C, _ = numeric.shape
        M, d = memory.shape[0], self.rank // self.heads
        q = self.query(self.numeric_norm(numeric)).reshape(B, C, self.heads, d).transpose(1, 2)
        memory = self.text_norm(memory.unsqueeze(0))
        k = self.key(memory).reshape(1, M, self.heads, d).transpose(1, 2)
        v = self.value(memory).reshape(1, M, self.heads, d).transpose(1, 2)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / d ** 0.5, dim=-1)
        read = torch.matmul(weights, v).transpose(1, 2).reshape(B, C, self.rank)
        return self.output(read)


def load_pca_memory(path, components, width):
    """[M, d] PCA memory written by pca.py."""
    memory = torch.load(path, map_location='cpu')
    if isinstance(memory, dict):
        memory = memory['dictionary']
    memory = torch.as_tensor(memory).float()
    if memory.shape == (width, components) and components != width:
        memory = memory.T  # CALF's original orientation
    if memory.shape != (components, width):
        raise ValueError('PCA memory {} has shape {}, expected [{}, {}]'.format(
            path, tuple(memory.shape), components, width))
    return memory.contiguous()


class Model(nn.Module):
    """STGF student: LoRA-adapted GPT-2 forecaster with a frozen memory interface.

    No teacher is loaded or executed here.  Teacher guidance only enters
    through the training loss, so the checkpoint alone is enough at inference.
    """

    def __init__(self, configs, device):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_memory = bool(getattr(configs, 'use_memory', 1))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=['c_attn'],
        )
        gpt_config = getattr(configs, 'gpt_config', None)
        if gpt_config is not None:
            # Architecture only (tests / restoring a complete checkpoint).
            gpt2 = AccustumGPT2Model(gpt_config)
        else:
            gpt2 = AccustumGPT2Model.from_pretrained(
                configs.gpt_path, output_attentions=True, output_hidden_states=True)
        gpt2.h = gpt2.h[:configs.gpt_layers]
        d_model = gpt2.config.n_embd
        self.gpt2 = get_peft_model(gpt2, peft_config)

        # Only LoRA, LayerNorm and positional parameters of the backbone are updated.
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = 'ln' in name or 'wpe' in name or 'lora' in name

        self.in_layer = NumericEncoder(configs.seq_len, d_model)
        self.out_layer = nn.Linear(d_model, configs.pred_len)

        if self.use_memory:
            memory = getattr(configs, 'pca_memory', None)
            if memory is None:
                memory = load_pca_memory(configs.word_embedding_path, configs.pca_components, d_model)
            self.register_buffer('pca_dictionary', memory)
            # A private random stream keeps the initialization of the encoder,
            # backbone and head identical with and without the interface.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(configs.cross_seed)
                self.text_cross = MemoryCrossAttention(d_model, configs.cross_rank, configs.cross_heads)
            self.text_gate = nn.Parameter(torch.zeros(()))

        self.to(device)

    def forecast(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        numeric = self.in_layer(x.transpose(1, 2))  # Z: [B, M, d]
        fused = numeric
        if self.use_memory:
            fused = numeric + self.text_gate.tanh() * self.text_cross(numeric, self.pca_dictionary)

        hidden, _ = self.gpt2(inputs_embeds=fused, use_cache=False,
                              output_hidden_states=False, output_attentions=False)
        outputs = self.out_layer(hidden + numeric).transpose(1, 2)  # [B, H, M]

        return {'outputs_time': outputs * stdev + means}

    def forward(self, x):
        return self.forecast(x[:, -self.seq_len:, :])
