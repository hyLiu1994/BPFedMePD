import h5py
import matplotlib.pyplot as plt
import numpy as np

# Re-define the file paths for the new set of uploaded files
# Mnist
file_paths = [
    # "./results/Mnist_small_FedSI_p_0.1_1.0_0.0001_10u_50b_20_0_0.01.h5",
    "./results/Mnist_small_FedSI_p_0.1_1.0_0.0001_10u_50b_20_0_0.03.h5",
    "./results/Mnist_small_FedSI_p_0.1_1.0_0.0001_10u_50b_20_0_0.05.h5",
    "./results/Mnist_small_FedSI_p_0.1_1.0_0.0001_10u_50b_20_0_0.07.h5",
    "./results/Mnist_small_FedSI_p_0.1_1.0_0.0001_10u_50b_20_0_0.09.h5"
]

# Cifar-10
file_paths = [
    # "./results/Cifar10_small_FedSI_0.01_1.0_0.0001_10u_50b_20_0_0.01.h5",
    #"./results/Cifar10_small_FedSI_0.01_1.0_0.0001_10u_50b_20_0_0.03.h5",
    #"./results/Cifar10_small_FedSI_0.01_1.0_0.0001_10u_50b_20_0_0.05.h5",
    #"./results/Cifar10_small_FedSI_0.01_1.0_0.0001_10u_50b_20_0_0.07.h5",
    #"./results/Cifar10_small_FedSI_0.01_1.0_0.0001_10u_50b_20_0_0.09.h5"
]



# Define a function to extract the accuracy data from the .h5 files
def extract_accuracy_data(file_paths):
    accuracies = []
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as file:
            accuracies.append(np.array(file['rs_glob_acc'][:]) * 100)  # Convert to percentage
    return accuracies

# Extract the accuracies from all files
all_accuracies = extract_accuracy_data(file_paths)

# Adjust the plot to display only epochs 100 to 800 and label the lines with the actual file names instead of "File 1 - File 5"

# Extract the file names from the paths for labeling
file_labels = [path.split('/')[-1].replace('.h5', '') for path in file_paths]

# Now, let's plot the accuracies again with the specified range and labels
plt.figure(figsize=(14, 7))
for i, accuracy in enumerate(all_accuracies):
    plt.plot(range(10, 800), accuracy[10:800], label=file_labels[i])
plt.title('Model Global Accuracy Between Epochs 100 and 800')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("./test.png")


def smooth_curve(points, factor=0.1):
    """Apply exponential moving average to smooth the curve."""
    smoothed_points = np.zeros_like(points)
    smoothed_points[0] = points[0]
    for i in range(1, len(points)):
        smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
    return smoothed_points

# Apply smoothing to each accuracy series
smoothed_accuracies = [smooth_curve(accuracy) for accuracy in all_accuracies]

# Define a list of distinct colors
distinct_colors = ['blue', 'green', 'red', 'purple', 'orange']

# Plotting the smoothed accuracies
plt.figure(figsize=(14, 7), dpi=120)
for i, accuracy in enumerate(smoothed_accuracies):
    epoch_range = range(10, min(800, len(accuracy)))
    plt.plot(epoch_range, accuracy[10:min(800, len(accuracy))], label=file_labels[i], color=distinct_colors[i])

plt.title('Smoothed Model Global Accuracy Between Epochs 100 and 800', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("./test.png")
