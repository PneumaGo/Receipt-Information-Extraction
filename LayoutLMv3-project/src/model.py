import torch
import pytorch_lightning as pl
from transformers import LayoutLMv3ForTokenClassification, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score

class LayoutLMv3Module(pl.LightningModule):
    def __init__(self, label2id, lr=2e-5):
        """
        Initializes the LightningModule for LayoutLMv3 Token Classification.
        
        Args:
            label2id: Dictionary mapping labels to IDs.
            lr: Base learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        
        # Load the pre-trained LayoutLMv3 model
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base", 
            num_labels=len(label2id) 
        )
        # Buffer to store validation outputs for epoch-end metric calculation
        self.validation_step_outputs = []

    def forward(self, batch):
        """Forward pass through the model."""
        return self.model(**batch)
    
    def training_step(self, batch, batch_idx):
        """Executes a single training step and logs loss."""
        outputs = self(batch)
        loss = outputs.loss 
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Executes a validation step, processes predictions, and handles masks."""
        outputs = self(batch)
        val_loss = outputs.loss
        
        # Move to CPU and convert to numpy for metric calculation
        preds = outputs.logits.argmax(-1).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        
        true_predictions, true_labels = [], []
        for i in range(len(labels)):
            # Filter out special tokens (like padding/subwords) using mask -100
            mask = labels[i] != -100 
            true_predictions.append([self.id2label[p] for p in preds[i][mask]])
            true_labels.append([self.id2label[l] for l in labels[i][mask]])
            
        self.log("val_loss", val_loss, prog_bar=True)
        output = {"preds": true_predictions, "labels": true_labels}
        self.validation_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
        """Calculates and logs aggregate metrics (F1, Precision, Recall) at the end of the epoch."""
        if not self.validation_step_outputs: return
        
        # Flatten predictions and labels across all validation batches
        all_preds = [p for out in self.validation_step_outputs for p in out["preds"]]
        all_labels = [l for out in self.validation_step_outputs for l in out["labels"]]
        
        # Calculate sequence labeling metrics
        val_f1 = f1_score(all_labels, all_preds)
        self.log("val_f1", val_f1, prog_bar=True) 
        self.log("val_precision", precision_score(all_labels, all_preds))
        self.log("val_recall", recall_score(all_labels, all_preds))
        
        # Clear buffer for the next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Sets up the AdamW optimizer with differential learning rates and weight decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        
        # Group parameters to apply different LR or Weight Decay
        optimizer_grouped_parameters = [
            {
                # Standard parameters with weight decay
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n],
                "weight_decay": 0.01,
                "lr": self.hparams.lr
            },
            {
                # Bias and LayerNorm parameters without weight decay
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n],
                "weight_decay": 0.0,
                "lr": self.hparams.lr
            },
            {
                # Classifier head with a higher learning rate (10x)
                "params": [p for n, p in self.model.classifier.named_parameters()],
                "weight_decay": 0.01,
                "lr": self.hparams.lr * 10 
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        # Setup learning rate scheduler with linear warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            }
        }