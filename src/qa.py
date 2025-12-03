from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

from typing import List, Dict
from .config import QA_MODEL_NAME

class QASystem:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        self.model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def answer_from_contexts(self, question: str, contexts: List[str]) -> Dict:
        # Simple: concatenate top contexts into one big context
        context = "\n\n".join(contexts)

        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1

        answer_ids = inputs["input_ids"][0][start_idx:end_idx]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        return {
            "answer": answer,
            "context_used": context
        }
