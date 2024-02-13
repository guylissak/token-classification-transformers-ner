import matplotlib.pyplot as plt
from typing import List

def plot_loss_and_accuracy_curves(training_loss: List, test_loss: List, accuracy: List):
  """
  Plot loss and accuracy curves
  """
  # Plot training & validation loss values
  epochs = range(1, len(training_loss) + 1)
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, training_loss, 'b', label='Training loss', marker='o')
  plt.plot(epochs, test_loss, 'r', label='Test loss', marker='o')
  plt.title('Training and Test Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  # Plot accuracy values
  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, 'g', label='Accuracy')
  plt.title('Training Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.show()
