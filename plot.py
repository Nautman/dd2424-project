import matplotlib.pyplot as plt

# Data for training and validation accuracy
epochs = list(range(1, 13))
training_accuracy = [0.3995, 0.8376, 0.8862, 0.9066, 0.9151, 0.9117, 0.9222, 0.9276, 0.9290, 0.9327, 0.9351, 0.9395]
validation_accuracy = [0.7853, 0.8764, 0.8995, 0.9049, 0.9117, 0.9158, 0.9158, 0.9158, 0.9171, 0.9158, 0.9226, 0.9212]

# Data for center cropping accuracy
center_cropping_accuracy = [0.4147, 0.8224, 0.8692, 0.8920, 0.9032, 0.9042, 0.9127, 0.9209, 0.9253, 0.9293, 0.9314, 0.9361]
center_cropping_validation_accuracy = [0.7717, 0.8546, 0.8954, 0.8967, 0.9090, 0.9090, 0.9076, 0.9144, 0.9171, 0.9198, 0.9194, 0.9227]

# Plotting the training and validation accuracy with center cropping
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracy, label='Baseline Training Accuracy', marker='o')
plt.plot(epochs, validation_accuracy, label='Baseline Validation Accuracy', marker='x')
plt.plot(epochs, center_cropping_accuracy, label='Random resize crop Training Accuracy', marker='o', linestyle='--')
plt.plot(epochs, center_cropping_validation_accuracy, label='Random resize crop Validation Accuracy', marker='x', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs (Baseline vs. Random resize crop)')
plt.legend()
plt.grid(True)
plt.show()
