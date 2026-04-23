import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import numpy as np
import albumentations as A
from difflib import SequenceMatcher

class SROIEDataset(Dataset):
    def __init__(self, folder_path, processor, label_map, train=True):
        self.processor = processor 
        self.label_map = label_map 
        self.root = Path(folder_path)
        self.train = train
        
        # Define image augmentation pipeline for training
        self.transform = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=5, border_mode=0, value=(255, 255, 255), p=0.5), 
                A.Perspective(scale=(0.02, 0.05), pad_val=(255, 255, 255), p=0.5), 
            ], p=0.4),
        
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.5), 
                A.MedianBlur(blur_limit=3, p=0.5), 
                A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5), 
            ], p=0.3),
        
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.RandomShadow(p=0.5), 
                A.CLAHE(clip_limit=2.0, p=0.5), 
            ], p=0.3),
        
            A.ShiftScaleRotate(
                shift_limit=0.03, scale_limit=0.05, rotate_limit=2, 
                border_mode=0, value=(255, 255, 255), p=0.3
            ),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        
        self.file_ids = []
        img_dir = self.root / 'img' 

        # Filter files that have both bounding boxes and entity labels
        for img_file in img_dir.glob("*.jpg"):
            fid = img_file.stem
            if (self.root / 'box' / f"{fid}.txt").exists() and \
               (self.root / 'entities' / f"{fid}.txt").exists():
                self.file_ids.append(fid)

    def __len__(self):
        return len(self.file_ids)

    def _assign_label(self, text, entities):
        """Assigns an entity label to a specific word based on string matching."""
        text_clean = text.replace(",", "").strip().lower()
        
        for key, val in entities.items():
            val_clean = str(val).replace(",", "").strip().lower()
            
            # Exact or substring match
            if text_clean in val_clean or val_clean in text_clean:
                return key.upper()
            
            # Fuzzy match for OCR errors
            if SequenceMatcher(None, text_clean, val_clean).ratio() > 0.8:
                return key.upper()
        
        return "O" # Outside/Other
        

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        
        # Load image and ground truth entities
        img = Image.open(self.root / 'img' / f"{fid}.jpg").convert("RGB")
        with open(self.root / 'entities' / f"{fid}.txt", 'r') as f:
            entities = json.load(f)

        words, boxes, word_labels = [], [], []
        
        # Parse OCR results and bounding boxes
        with open(self.root / 'box' / f"{fid}.txt", 'r', errors='ignore') as f:
            for line in f.read().splitlines():
                if not line: continue
                
                parts = line.split(",")
                if len(parts) < 9: continue                        
                
                # Extract coordinates (x1, y1, x2, y2)
                coords = [float(parts[0]), float(parts[1]), float(parts[4]), float(parts[5])]
                
                # Join remaining parts as the text content
                text = ",".join(parts[8:])
                label = self._assign_label(text, entities)
                
                # Convert to BIO format (Beginning-Entity)
                final_label = f"B-{label}" if label != "O" else "O"
                
                words.append(text)
                boxes.append(coords)
                word_labels.append(self.label_map.get(final_label, 0))

        # Apply augmentations during training
        if self.train:
            img_np = np.array(img)
            word_indices = list(range(len(words)))
            
            try:
                augmented = self.transform(
                    image=img_np, 
                    bboxes=boxes, 
                    category_ids=word_indices
                )
                
                if len(augmented['bboxes']) > 0:
                    img = Image.fromarray(augmented['image'])
                    
                    new_words, new_word_labels = [], []
                    for i in augmented['category_ids']:
                        new_words.append(words[i])
                        new_word_labels.append(word_labels[i])
                    
                    words, word_labels = new_words, new_word_labels
                    boxes = augmented['bboxes']
            except Exception:
                pass # Fallback to original if augmentation fails

        # Normalize bounding boxes to 0-1000 scale for LayoutLM
        w, h = img.size
        normalized_boxes = []
        for box in boxes:
            normalized_boxes.append([
                max(0, min(1000, int(1000 * (box[0] / w)))),                    
                max(0, min(1000, int(1000 * (box[1] / h)))),                    
                max(0, min(1000, int(1000 * (box[2] / w)))),
                max(0, min(1000, int(1000 * (box[3] / h))))
            ])

        # Prepare final encoding using the LayoutLM processor
        encoding = self.processor(
            img, 
            words, 
            boxes=normalized_boxes, 
            word_labels=word_labels,
            truncation=True,        
            padding="max_length",   
            max_length=512,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by return_tensors="pt"
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        
        if "labels" in encoding:
            encoding["labels"] = encoding["labels"].to(torch.long)
            
        return encoding