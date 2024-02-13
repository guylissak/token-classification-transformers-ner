import numpy as np
from datasets import load_metric
from transformers import Trainer
from typing import Dict, Tuple, List

metric = load_metric("seqeval")

def compute_metrics(logits_and_labels) -> Dict:
  """ Computes precision, recall, f1 score and accuracy """

  logits, labels = logits_and_labels
  preds = np.argmax(logits, axis=-1)
  label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-JOBTITLE', 'I-JOBTITLE', 'B-IND', 'I-IND']

  # remove -100 from labels and predictions
  # and convert the label_ids to label names
  str_labels = [
    [label_names[t] for t in label if t != -100] for label in labels
  ]

  # do the same for predictions whenever true label is -100
  str_preds = [
    [label_names[p] for p, t in zip(pred, targ) if t != -100] \
      for pred, targ in zip(preds, labels)
  ]

  metrics = metric.compute(predictions=str_preds, references=str_labels)
  return {
    'precision': metrics['overall_precision'],
    'recall': metrics['overall_recall'],
    'f1': metrics['overall_f1'],
    'accuracy': metrics['overall_accuracy'],
  }

def get_epochs_metrics(trainer: Trainer) -> Tuple[List, List, List]:
  """
  Returns training loss, test loss, and accuracy
  """
  model_metrics = [(epoch.get("loss"), epoch.get("eval_loss"), epoch.get("eval_accuracy")) for epoch in trainer.state.log_history]
  training_loss = []
  test_loss = []
  accuracy = []
  for i, (train_loss, eval_loss, eval_accuracy) in enumerate(model_metrics):
    if train_loss:
      training_loss.append(float(f"{train_loss:.3f}"))
    if eval_loss:
      test_loss.append(float(f"{eval_loss:.3f}"))
    if eval_accuracy:
      accuracy.append(float(f"{eval_accuracy:.3f}"))

  return training_loss, test_loss, accuracy
  
