import matplotlib.pyplot as plt

# Epoch data
epochs = list(range(25))

# Training and validation loss
train_loss = [3.0531, 1.2742, 0.4765, 0.2875, 0.2152, 0.1563, 0.1208, 0.0951, 0.0812, 0.0654, 0.0592, 0.0471, 0.0409, 0.0349, 0.0291, 0.0278, 0.0261, 0.0239, 0.0208, 0.0191, 0.0169, 0.0145, 0.0171, 0.0142, 0.0131]
val_loss = [2.1617, 0.5730, 0.3464, 0.2576, 0.2329, 0.2181, 0.2029, 0.1888, 0.1861, 0.1914, 0.1882, 0.1866, 0.1834, 0.1879, 0.1861, 0.1818, 0.1777, 0.1862, 0.1829, 0.1881, 0.1860, 0.1806, 0.1890, 0.1847, 0.1779]

# Training and validation accuracy
train_acc = [0.4538, 0.8573, 0.9192, 0.9446, 0.9504, 0.9701, 0.9755, 0.9817, 0.9874, 0.9918, 0.9918, 0.9939, 0.9952, 0.9969, 0.9959, 0.9973, 0.9973, 0.9990, 0.9983, 0.9990, 0.9990, 0.9993, 0.9986, 0.9990, 0.9993]
val_acc = [0.8139, 0.9117, 0.9293, 0.9389, 0.9348, 0.9334, 0.9429, 0.9443, 0.9457, 0.9375, 0.9416, 0.9416, 0.9443, 0.9375, 0.9443, 0.9457, 0.9457, 0.9429, 0.9402, 0.9416, 0.9402, 0.9429, 0.9443, 0.9429, 0.9429]

# Plotting the loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting the accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
