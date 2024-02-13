""" Tokenization utils """
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from typing import List

#['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-JOBTITLE', 'I-JOBTITLE', 'B-IND', 'I-IND']
begin2inside = {
  1: 2,
  3: 4,
  5: 6,
  7: 8,
  9: 10
}

label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-JOBTITLE', 'I-JOBTITLE', 'B-IND', 'I-IND']

def align_targets(labels: List[int], word_ids: List[int]) -> List[int]:
  """
  Aligns token labels with their corresponding word IDs, adjusting labels for subword tokens.
  This function iterates over each token's word ID, generated during tokenization, to align each token
  with its appropriate label. Special tokens (like [CLS], [SEP], etc.) are assigned a label of -100,
  indicating they should be ignored during loss calculation. When a word is split into multiple subword tokens,
  all tokens from the same word are assigned the same label. If necessary, the function adjusts the label
  from 'B-' (beginning) to 'I-' (inside) for subsequent subword tokens of a word, based on a predefined
  mapping (`begin2inside`).
  """
  aligned_labels = []
  last_word = None
  for word in word_ids:
    if word is None:
      # it's a token like [CLS]
      label = -100
    elif word != last_word:
      # it's a new word!
      label = labels[word]
    else:
      # it's the same word as before
      label = labels[word]

      # change B-<tag> to I-<tag> if necessary
      if label in begin2inside:
        label = begin2inside[label]

    # add the label
    aligned_labels.append(label)

    # update last word
    last_word = word

  return aligned_labels

# tokenize both inputs and targets
def tokenize_fn(batch: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
  """
  Tokenizes input sequences and aligns NER labels with the tokenized inputs.

  This function applies a tokenizer to a batch of input sequences, handling both the tokenization
  and the alignment of NER labels with the resulting tokenized inputs. It ensures that each token
  has a corresponding label, appropriately handling special tokens (like [CLS], [SEP]) by assigning
  them a predefined ignore index (-100) and adjusting labels for subword tokens as necessary.

  The function iterates over each sequence in the batch, tokenizes the sequence, and then aligns
  the labels with the tokenized sequence using a custom label alignment function. The aligned labels
  are added to the tokenized inputs under the key 'labels'.
  """

  # tokenize the input sequence first
  # this populates input_ids, attention_mask, etc.
  tokenized_inputs = tokenizer(
    batch['tokens'], truncation=True, is_split_into_words=True
  )

  labels_batch = batch['ner_tags'] # original targets
  aligned_labels_batch = []
  for i, labels in enumerate(labels_batch):
    word_ids = tokenized_inputs.word_ids(i)
    aligned_labels_batch.append(align_targets(labels, word_ids))

  # the 'target' must be stored in key called 'labels'
  tokenized_inputs['labels'] = aligned_labels_batch

  return tokenized_inputs
