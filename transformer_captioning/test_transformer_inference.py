import torch
from utils import * 
from torch.utils.data import DataLoader
from trainer import Trainer
from transformer import TransformerDecoder
from matplotlib import pyplot as plt


def loss_func(predictions, labels):
        #TODO - Compute cross entropy loss between predictions and labels. 
        #Make sure to compute this loss only for indices where label is not the null token.
        #The loss should be averaged over batch and sequence dimensions.

        # Create a mask to mask out the null token losses (idx for null token is 0)
        null_mask = labels.clone()
        null_mask[null_mask != 0] = 1

        # Calculate the loss
        loss = (torch.nn.functional.cross_entropy(predictions.permute(0,2,1), labels, reduction='none')*null_mask).sum() / null_mask.sum()

        return loss

if __name__ == "__main__":
    set_all_seeds(42) ### DO NOT CHANGE THIS LINE
    exp_name = 'case1'

    train_dataset = CocoDataset(load_coco_data(max_train=1024), 'train')
    train_dataloader =  DataLoader(train_dataset, batch_size=64)

    val_dataset = CocoDataset(load_coco_data(max_val = 1024), 'val')
    val_dataloader =  DataLoader(val_dataset, batch_size=64)

    features, captions = next(iter(train_dataloader))[0:2]

    transformer = TransformerDecoder(
          word_to_idx=train_dataset.data['word_to_idx'],
          idx_to_word = train_dataset.data['idx_to_word'],
          input_dim=train_dataset.data['train_features'].shape[1],
          embed_dim=256,
          num_heads=2,
          num_layers=2,
          max_length=30,
          device = 'cpu'
    )

    logits = transformer(features, captions[:, :-1])
    print(logits.shape)

    # Test the loss
    loss = loss_func(logits, captions[:, 1:])
    print(loss)
    

