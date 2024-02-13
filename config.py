import os
import torch

class Config:

    DATASET_PATH = "labels.json"
    MODELS_OUTPUT_PATH = "finetuned_model_weights_ner_bert_guy.pth"
    MODEL_CHECKPOINT = "bert-base-cased"
    # Hyper parameters
    INIT_LR = 0.00002
    NUM_EPOCHS = 8
    BATCH_SIZE = 32
    WEIGHT_DECAY = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()
