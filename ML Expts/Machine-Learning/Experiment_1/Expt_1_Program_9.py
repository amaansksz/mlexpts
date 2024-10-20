import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'epoch': np.arange(1, 21),
    'train_loss': np.random.uniform(0.2, 0.6, 20),
    'val_loss': np.random.uniform(0.3, 0.7, 20),
    'train_accuracy': np.random.uniform(0.7, 0.95, 20),
    'val_accuracy': np.random.uniform(0.6, 0.9, 20)
}
df = pd.DataFrame(data)
df = df.sort_values('epoch')
plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1)
plt.plot(df['epoch'],df['train_loss'] ,label = "Train Loss", color = "orange", marker ="o" )
plt.plot(df['epoch'],df['val_loss'] ,label = "Value Loss", color = "blue", marker ="o" )
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Visualization By Amaan Shaikh S. 211P052')
plt.legend()


# Plot 2
plt.subplot(1, 2, 2)
plt.plot(df['epoch'],df['train_accuracy'] ,label = "Train Accuracy", color = "yellow", marker ="o" )
plt.plot(df['epoch'],df['val_accuracy'] ,label = "Value Accuracy", color = "red", marker ="o" )
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training accuracy Visualization By Amaan Shaikh S. 211P052')
plt.legend()
plt.show()