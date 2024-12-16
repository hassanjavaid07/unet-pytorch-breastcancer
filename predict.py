import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.helper_functions import calculateDiceScore



# Implements functionlity for real-time image preprocessing and class prediction
def preprocessImage(image_path, image_size=128):
    img = Image.open(image_path).convert('L')

    # Define transformations to be applied to the image
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),    # Convert to grayscale
        transforms.Resize((image_size, image_size)),    # Resize to image_size
        transforms.ToTensor(),                          # Convert to tensor
    ])

    # Apply transformations to the image
    img_processed = transform(img).unsqueeze(0) # Add batch dimension

    return img_processed

def predictImageMask(image_path, original_mask_path, model, threshold=0.5):
    # Preprocess the image
    image = preprocessImage(image_path)
    original_mask = preprocessImage(original_mask_path)
    model.eval()

    # Perform prediction
    with torch.no_grad():
        predicted_mask = model(image)
        dice_score = calculateDiceScore(predicted_mask, original_mask)
        print(f"Dice score for input image: {dice_score}")
        fig, axes = plt.subplots(1, 3, figsize=(15, 3))

        image_np = image.cpu().detach().numpy().squeeze(0)
        original_mask_np = original_mask.cpu().detach().numpy().squeeze(0)
        predicted_mask_np = predicted_mask.cpu().detach().numpy()
        # print(image_np.shape)
        
        # Original Image
        axes[0].imshow(np.transpose(image_np, (1, 2, 0)))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Original Mask
        axes[1].imshow(original_mask_np.squeeze(), cmap='gray')
        axes[1].set_title('Original Mask')
        axes[1].axis('off')

        # Predicted Mask
        predicted_mask_np = (predicted_mask_np > threshold)
        axes[2].imshow(predicted_mask_np.squeeze(), cmap='gray')
        axes[2].set_title(f'Predicted Mask with Dice Score: {dice_score:.4f}')
        axes[2].axis('off')

        plt.suptitle("Image Mask Prediction")
        plt.tight_layout()
        plt.draw()

