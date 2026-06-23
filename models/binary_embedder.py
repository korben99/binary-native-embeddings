import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .ste import binarize


class BinaryEmbedder(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-mini", binary_dim=4096):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 256 for bert-mini
        self.projection = nn.Sequential(
            nn.Linear(hidden, binary_dim),
            nn.LayerNorm(binary_dim),
        )
        self.binary_dim = binary_dim

    def _mean_pool(self, token_embs, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, binarize_output=True):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self._mean_pool(out, attention_mask)
        projected = self.projection(pooled)          # pre-binarization logits
        if binarize_output:
            return binarize(projected)               # {0,1}^4096
        return projected                             # float, for loss computation

    def encode(self, texts, tokenizer, device="cpu", batch_size=64):
        self.eval()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                embs = self.forward(enc["input_ids"], enc["attention_mask"], binarize_output=True)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)


def binary_contrastive_loss(anchors_logits, positives_logits, temperature=0.05):
    """
    Loss on pre-binarization logits via tanh + cosine.
    tanh maps to (-1,+1), aligned with the {-1,+1} STE output so the training
    signal directly optimizes what the eval metric measures.
    """
    a = torch.tanh(anchors_logits)
    p = torch.tanh(positives_logits)
    a_norm = F.normalize(a, dim=-1)
    p_norm = F.normalize(p, dim=-1)
    sim = torch.mm(a_norm, p_norm.T) / temperature
    labels = torch.arange(len(a), device=anchors_logits.device)
    return nn.CrossEntropyLoss()(sim, labels)
