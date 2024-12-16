import torch



# Implements saving and loading of the model
def saveModel(model_state_dict, filename):
    # if torch.cuda.is_available():
    #     model.to('cpu')  # Move model to CPU before saving if it's on GPU
    torch.save(model_state_dict, filename)



def loadModel(model, filename):
    if torch.cuda.is_available():
        # Load model on CPU first and then move it to GPU if available
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        model.to('cuda')
    else:
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    return model




# Implements Dice Score calculation function
def calculateDiceScore(pred_mask, target_mask, eps=1e-7):
    assert pred_mask.size() == target_mask.size()
    intersection = torch.sum(pred_mask * target_mask)
    union = torch.sum(pred_mask) + torch.sum(target_mask)
    dice_score = (2. * intersection + eps) / (union + eps)
    return dice_score



# Implements Dice Loss calculation function
def calculateDiceLoss(pred_mask, target_mask, eps=1e-7):
    return 1 - calculateDiceScore(pred_mask, target_mask)



# Implements the movement of tensor to CPU and convert to NumPy array
def convertToNumpy(tensor):
    return tensor.cpu().detach().numpy()



# Create custom dataset with specified number of images from each class
def genSubsetData(dataloader, numImages=50):
    class_indices = {0: 0, 1: 0, 2: 0}
    subset_data = []

    for images, masks, labels in dataloader:
        for image, mask, label in zip(images, masks, labels):
            if label.item() in class_indices and class_indices[label.item()] < numImages:
                subset_data.append((image, mask, label))
                class_indices[label.item()] += 1
            if all(count >= numImages for count in class_indices.values()):
                break
        if all(count >= numImages for count in class_indices.values()):
            break

    return subset_data
