import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class FloatEmbedder(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-mini", output_dim=384):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 256 for bert-mini
        # project to target dim so memory footprint matches the plan
        self.projection = nn.Linear(hidden, output_dim) if hidden != output_dim else nn.Identity()
        self.output_dim = output_dim

    def _mean_pool(self, token_embs, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self._mean_pool(out, attention_mask)
        return self.projection(pooled)

    def encode(self, texts, tokenizer, device="cpu", batch_size=64):
        self.eval()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                embs = self.forward(enc["input_ids"], enc["attention_mask"])
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)


def mnrl_loss(anchors, positives, temperature=0.05):
    """MultipleNegativesRankingLoss — treats other batch items as negatives."""
    a = F.normalize(anchors, dim=-1)
    p = F.normalize(positives, dim=-1)
    sim = torch.mm(a, p.T) / temperature
    labels = torch.arange(len(a), device=anchors.device)
    return nn.CrossEntropyLoss()(sim, labels)
