import torch
from utils.helper_functions import calculateDiceScore


# Implements the test function to evaluate our trained model
def evaluateModel(model, test_loader):
    model.eval()
    correct_preds = []
    wrong_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Testing Loop
    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)

            # Generate predicted masks from test images
            predicted_masks = model(images)

            # Calculate dice score over each batch image and identify correct/incorrect predictions
            for image_idx in range(len(images)):
                dice_score = calculateDiceScore(predicted_masks[image_idx], masks[image_idx])
                if dice_score > 0.6:
                    # Correct pred
                    correct_preds.append((batch_idx, image_idx, images[image_idx], masks[image_idx], predicted_masks[image_idx], dice_score))
                else:
                    # Wrong pred
                    wrong_preds.append((batch_idx, image_idx, images[image_idx], masks[image_idx], predicted_masks[image_idx], dice_score))
    return (correct_preds, wrong_preds)

