import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from helper_functions import convertToNumpy



# ==========================================================================================
# Implements trained_model plotting and visualization functions on train/valid/test datasets
# ==========================================================================================


# Implements plotting of train and validation dataset loss and accuracy curves
def plotTrainValidHistory(train_loss, valid_loss, train_acc, valid_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train', color='blue', linestyle='-')
    plt.plot(epochs, valid_loss, label='Validation', color='orange', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.ylim(0.05, 2.05)
    plt.grid(True)
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train', color='blue', linestyle='-')
    plt.plot(epochs, valid_acc, label='Validation', color='orange', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Curve')
    plt.ylim(0.05, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.draw()




# Implements visualization of test predictions
def visualizePredictions(predictions, test_loader, num_samples=5, threshold=0.5):
    num_cols = 3
    num_rows = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3*num_rows))

    selected_samples = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)

    for row_idx in range(num_rows):
        idx = selected_samples[row_idx]
        # print(idx)
        batch_idx, image_idx, image, original_mask, predicted_mask, dice_score = predictions[idx]

        image_np, original_mask_np, predicted_mask_np = map(convertToNumpy, (image, original_mask,
                                                                    predicted_mask))
        # Original Image
        axes[row_idx, 0].imshow(np.transpose(image_np, (1, 2, 0)))
        axes[row_idx, 0].set_title('Original Image')
        axes[row_idx, 0].axis('off')

        # Original Mask
        axes[row_idx, 1].imshow(original_mask_np.squeeze(), cmap='gray')
        axes[row_idx, 1].set_title('Original Mask')
        axes[row_idx, 1].axis('off')

        # Predicted Mask
        predicted_mask_np = (predicted_mask_np > threshold)
        axes[row_idx, 2].imshow(predicted_mask_np.squeeze(), cmap='gray')
        axes[row_idx, 2].set_title(f'Predicted Mask with Dice Score: {dice_score:.4f}')
        axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.draw()




# Implements plotting of sample images from dataloader
def plotSampleImage(dataloader, suptitle, oneHot=False):
    dataset = dataloader.dataset
    random_idx = random.randint(0, len(dataset) - 1)
    img, mask, label = dataset[random_idx]
    transform = transforms.Grayscale()

    if int(label) == 0:
        title = "Normal Class"
    elif int(label) == 1:
        title = "Benign Class"
    else:
        title = "Malignant Class"

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(transform(img).squeeze(), cmap="gray")
    ax.set_title(f"{title}, Label: {label}", fontsize=14)
    ax.axis('off')

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(transform(mask).squeeze(), cmap="gray")
    ax.set_title(f"{title}, Label: {label}", fontsize=14)
    ax.axis('off')

    fig.suptitle(suptitle, fontsize=16, fontweight='bold', color='blue')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()
