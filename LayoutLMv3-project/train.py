import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoProcessor
from data_module import SROIEDataModule
from model_module import LayoutLMv3Module

# --- Training Callbacks Configuration ---

# Saves the best model based on Validation F1 score
checkpoint_callback = ModelCheckpoint(
    monitor="val_f1",
    dirpath="checkpoints",
    filename="best-layoutlmv3-sroie",
    mode="max",
    save_top_k=1
)

# Stops training if Validation F1 does not improve for 4 consecutive epochs
early_stop_callback = EarlyStopping(
    monitor="val_f1", 
    patience=4, 
    mode="max"
)

# --- Label Mapping & Processor ---

label2id = {
    "O": 0, 
    "B-COMPANY": 1, "I-COMPANY": 2, 
    "B-DATE": 3, "I-DATE": 4, 
    "B-ADDRESS": 5, "I-ADDRESS": 6, 
    "B-TOTAL": 7, "I-TOTAL": 8
}

# OCR is set to False because we are using the bounding boxes provided in the dataset
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
DATA_ROOT = "/kaggle/input/datasets/maxbegal/dataset-layoutlm/data"

# --- Data & Model Initialization ---

dm = SROIEDataModule(DATA_ROOT, processor, label2id, batch_size=4)
model = LayoutLMv3Module(label2id=label2id)

# --- Trainer Setup ---

trainer = pl.Trainer(
    max_epochs=20, 
    accelerator="gpu",
    devices=1,
    precision="16-mixed", # Uses Mixed Precision for faster training and lower VRAM usage
    logger=TensorBoardLogger("logs/", name="layoutlmv3_final"),
    callbacks=[checkpoint_callback, early_stop_callback]
)

# Start training process
trainer.fit(model, dm)

# --- Post-Training / Saving ---

# Load the best weights discovered during training
if checkpoint_callback.best_model_path:
    print(f"Loading best model from: {checkpoint_callback.best_model_path}")
    model = LayoutLMv3Module.load_from_checkpoint(
        checkpoint_callback.best_model_path, 
        label2id=label2id
    )

# Export the final model and processor for inference
model.model.save_pretrained("layoutlmv3_sroie_final")
processor.save_pretrained("layoutlmv3_sroie_final")

print("✅ Training complete. Model saved to 'layoutlmv3_sroie_final'")