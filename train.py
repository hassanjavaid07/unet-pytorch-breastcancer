import torch
import wandb
import argparse
import logging
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from unet_model.unet_model import UNetModel
from utils.dataloader_utils import make_data
from utils.helper_functions import convertToNumpy, calculateDiceScore, calculateDiceLoss




# Implements training function with early stopping and learning rate decay
def trainModel(model, train_loader, val_loader, lr, num_epochs, 
               max_early_stop, patience, momentum, weight_decay, 
               device, args, save_checkpoint=True, dir_checkpoint=Path('./checkpoints/')):
#     STEPS:
#     1. Define loss-criterion and optimizer
#     2. Initialize logging
#     3. Define lists for storing training & validation loss and accuracy history
#     4. Use BCE/CrossEntropy and Dice Loss as loss function
#     5. Initailize learning rate decay scheduler and early_stopping_counter.
#     6. Calculate total loss as sum of BCE loss and dice-loss.
#     7. Iterate over train_loader & val_loader and calculate total loss and dice-score for each sample
#     8. If valid_loss is less than best_valid_loss, increase early_stopping_counter.
#     9. Update learning rate by decay factor using scheduler step in order to maximize dice-score.
#     10. Store the results in relevant lists.
#     11. Save the best model as defined by epoch_val_score.
#     12. Finish training after NUM_EPOCHS
#     13. Return Training/Validation loss and dice history

    
    # Define loss-criterion and optimizer
    criterion = nn.BCEWithLogitsLoss() if model.n_classes == 1 else nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr,
                            weight_decay=weight_decay, momentum=momentum,
                            foreach=True)


    # Initialize logging
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(num_epochs=num_epochs, batch_size=args.batch_size, learning_rate=lr,
    #          val_size=args.val_size, save_checkpoint=save_checkpoint)
    # )
    
    logging.info(f'''Training & validation loop started:
        Epochs:          {num_epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {lr}
        Training size:   {1.0 - args.val_size - args.test_size:.1f}
        Validation size: {args.val_size}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    
    model.to(device)

    best_valid_loss = float('inf')
    early_stopping_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)
    
    train_dice_loss_history = []
    train_dice_score_history = []
    val_dice_loss_history = []
    val_dice_score_history = []

    total_train_samples = 0
    total_val_samples = 0
    best_dice = 0.0

    total_train_samples = sum(images.size(0) for images, _, _ in train_loader)
    logging.info(f"Total Train Samples = {total_train_samples}")

    total_val_samples = sum(images.size(0) for images, _, _ in val_loader)
    logging.info(f"Total Validation Samples = {total_val_samples}")

    for epoch in range(num_epochs):
        
        # Training loop
        model.train()
        train_dice_loss = 0.0
        train_dice_score = 0.0

        for images, true_masks, _ in train_loader:
            images, true_masks = images.to(device), true_masks.to(device)
            optimizer.zero_grad()
            pred_masks = model(images)

            loss = criterion(pred_masks, true_masks)
            loss += calculateDiceLoss(pred_masks, true_masks)

            train_dice_loss += loss.item()
            train_dice_score += calculateDiceScore(pred_masks, true_masks)

            loss.backward()
            optimizer.step()

        epoch_train_score = train_dice_score / len(train_loader)
        epoch_train_loss = train_dice_loss / len(train_loader)
        train_dice_score_history.append(convertToNumpy(epoch_train_score))
        train_dice_loss_history.append(epoch_train_loss)

        # Validation loop
        model.eval()
        val_dice_loss = 0.0
        val_dice_score = 0.0
        with torch.no_grad():
            for images, true_masks, _ in val_loader:
                images, true_masks = images.to(device), true_masks.to(device)
                pred_masks = model(images)

                loss = criterion(pred_masks, true_masks)
                loss += calculateDiceLoss(pred_masks, true_masks)
                val_dice_loss += loss.item()

                val_dice_score += calculateDiceScore(pred_masks, true_masks)


        epoch_val_score = val_dice_score / len(val_loader)
        epoch_val_loss = val_dice_loss / len(val_loader)
        val_dice_score_history.append(convertToNumpy(epoch_val_score))
        val_dice_loss_history.append(epoch_val_loss)
        logging.info(f'''
        	Epoch:			    {epoch+1}/{num_epochs} 
        	Train Loss:		    {epoch_train_loss:.4f} 
        	Train Dice Score:	{epoch_train_score:.4f} 
        	Valid Loss:		    {epoch_val_loss:.4f} 
        	Valid Dice Score:	{epoch_val_score:.4f}
        ''')

        # Save best model
        if epoch_val_score > best_dice:
            best_dice = epoch_val_score
            best_epoch = epoch + 1
            best_model = model.state_dict()

        # Early stopping
        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= max_early_stop:
                logging.info("Early stopping triggered!")
                if save_checkpoint:
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict = model.state_dict()
                    torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                    logging.info(f'Checkpoint {epoch} saved!')
                return best_model, best_dice, best_epoch, train_dice_loss_history, val_dice_loss_history, train_dice_score_history, val_dice_score_history
                break

        # Learning rate decay
        scheduler.step(epoch_val_score)

    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        logging.info(f'Checkpoint {epoch} saved!')

    logging.info("Training finished.")
    return best_model, best_dice, best_epoch, train_dice_loss_history, val_dice_loss_history, train_dice_score_history, val_dice_score_history



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, required=True, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--early-stopping-factor', '-es', metavar='ES', dest='early_stopping', 
                        type=int, default=10, help='Early stopping factor')
    parser.add_argument('--momentum', '-m', metavar='M', dest='momentum', type=float,
                        default=0.999, help='Momentum')
    parser.add_argument('--threshold', '-th', metavar='TH', dest='threshold', type=float, 
                        default=0.5, help='Threshold')
    parser.add_argument('--patience', '-p', metavar='P', dest='patience', type=int, 
                        default=10, help='Patience')
    parser.add_argument('--weight-decay', '-wd', metavar='WD', dest='weight_decay', type=float, 
                        default=1e-6, help='Weight decay')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--validation', '-v', dest='val_size', type=float, default=0.2,
                        help='Percent of data that is used as validation (0.0-1.0)')
    parser.add_argument('--test', '-t', dest='test_size', type=float, default=0.1,
                        help='Percent of data that is used as test (0.0-1.0)')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--random-seed', '-rs', metavar='RS', dest='random_seed', type=int,
                        default=42, help='Random seed')
    
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}\n')

    ROOT_DIR = "./data"
    train_loader, val_loader, _ = make_data(ROOT_DIR, args)

    # Change here to adapt to your data
    # n_channels=1 for Grayscale images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNetModel(n_channels=1, n_classes=args.classes)
    
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        trainModel(model=model,
                   train_loader=train_loader, 
                   val_loader=val_loader,
                   lr=args.lr, 
                   num_epochs=args.epochs, 
                   max_early_stop=args.early_stopping, 
                   patience=args.patience,
                   momentum=args.momentum,
                   weight_decay=args.weight_decay, 
                   device=device,
                   args=args)
    
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        trainModel(model=model,
                   train_loader=train_loader, 
                   val_loader=val_loader,
                   lr=args.lr, 
                   num_epochs=args.epochs, 
                   max_early_stop=args.early_stopping, 
                   patience=args.patience,
                   momentum=args.momentum,
                   weight_decay=args.weight_decay, 
                   device=device,
                   args=args)
