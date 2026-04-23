import pytorch_lightning as pl
from torch.utils.data import DataLoader
import SROIEDataset

class SROIEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, processor, label_map, batch_size=4):
        """
        Initializes the DataModule for the SROIE dataset.
        
        Args:
            data_dir: Path to the root directory containing train/val/test splits.
            processor: The LayoutLM processor (image + text).
            label_map: Dictionary mapping string labels to integer IDs.
            batch_size: Number of samples per batch.
        """
        super().__init__()
        self.data_dir = data_dir   
        self.processor = processor 
        self.label_map = label_map 
        self.batch_size = batch_size 
    
    def setup(self, stage=None):
        """
        Splits and prepares datasets based on the current stage (fit, test, etc.).
        """
        # Set up training and validation datasets
        if stage == "fit" or stage is None:
            self.train_ds = SROIEDataset(
                f"{self.data_dir}/train", self.processor, self.label_map, train=True
            )
            self.val_ds = SROIEDataset(
                f"{self.data_dir}/val", self.processor, self.label_map, train=False
            )
        
        # Set up test dataset
        if stage == "test" or stage is None:
            self.test_ds = SROIEDataset(
                f"{self.data_dir}/test", self.processor, self.label_map, train=False
            )

    def train_dataloader(self):
        """Returns the training data loader with shuffling enabled."""
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,         
            num_workers=2,         
            pin_memory=True       
        )

    def val_dataloader(self):
        """Returns the validation data loader."""
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,         
            num_workers=2,
            pin_memory=True
        )

    def test_dataloader(self):
        """Returns the test data loader."""
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )