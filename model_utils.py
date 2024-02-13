""" Model utils"""
from transformers import BertForTokenClassification, BertTokenizerFast, \
 pipeline, AutoModelForTokenClassification, Pipeline, PreTrainedModel
import torch
from config import config as conf
from typing import Dict

def save_model(model: PreTrainedModel, model_path: str):
  """ Save just the model parameters (weights) to a file """
  torch.save(model.state_dict(), model_path)
  print(f"Model saved to {model_path}")
  

def load_model(model_path: str, id2label: Dict, label2id: Dict, 
               checkpoint: str = "bert-base-cased") -> Pipeline:
  """"
  Loads a fine-tuned token classification model and its tokenizer to create a named entity recognition (NER) pipeline.
  This function initializes a model architecture based on a specified checkpoint, loads fine-tuned weights from a given path,
  and sets up a Hugging Face pipeline for token classification tasks, such as NER. The pipeline created by this function
  is ready for making predictions on new data.
    """
               
  model = AutoModelForTokenClassification.from_pretrained(
      checkpoint,
      id2label=id2label,
      label2id=label2id,
  )

  # Load your fine-tuned model weights
  model.load_state_dict(torch.load(model_path))

  # Ensure the model is in evaluation mode
  model.eval()


  # Load tokenizer
  tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

  # Create the pipeline using your fine-tuned model and tokenizer
  ner_pipeline = pipeline(
      "token-classification",
      model=model,
      tokenizer=tokenizer,
      aggregation_strategy="simple"
  )

  print("Model loaded successfully, NER pipeline is up! :-), enjoy!!!!")

  return ner_pipeline
