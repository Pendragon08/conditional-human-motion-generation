"""Optional actual DistilBERT interface; never replaces text with random tokens."""
import torch
from torch import nn

class FrozenDistilBert(nn.Module):
    def __init__(self,model_path,local_files_only=True):
        super().__init__()
        from transformers import AutoTokenizer,DistilBertModel
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=local_files_only)
        self.encoder=DistilBertModel.from_pretrained(model_path,local_files_only=local_files_only)
        self.encoder.requires_grad_(False);self.encoder.eval()
        if self.encoder.config.dim!=768:raise ValueError('Paper requires 768-dimensional DistilBERT tokens')

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    @torch.no_grad()
    def forward(self,texts):
        inputs=self.tokenizer(texts,padding=True,truncation=True,return_tensors='pt')
        device=next(self.encoder.parameters()).device
        inputs={k:v.to(device) for k,v in inputs.items()}
        return self.encoder(**inputs).last_hidden_state,inputs['attention_mask'].bool()
