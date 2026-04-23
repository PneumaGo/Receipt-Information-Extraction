import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re
from pathlib import Path

def run_inference(image_path, box_file_path, model, processor, label2id, threshold=0.85):
    """
    Runs model inference on a single image and filters results using heuristics.
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    id2label = {v: k for k, v in label2id.items()}
    
    words, boxes = [], []
    try:
        with open(box_file_path, 'r', errors='ignore') as f:
            for line in f:
                parts = line.strip().split(',', 8)
                if len(parts) < 9: continue
                # Extract coordinates and text content
                raw_box = [int(parts[0]), int(parts[1]), int(parts[4]), int(parts[5])]
                words.append(parts[8])
                boxes.append(raw_box)
    except FileNotFoundError:
        return [], [], []

    # Normalize boxes to 0-1000 range as required by LayoutLMv3
    normalized_boxes = [[int(1000*(b[0]/w)), int(1000*(b[1]/h)), int(1000*(b[2]/w)), int(1000*(b[3]/h))] for b in boxes]
    
    # Prepare inputs for the model
    encoding = processor(image, words, boxes=normalized_boxes, return_tensors="pt")
    word_ids = encoding.word_ids(batch_index=0)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in encoding.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply Softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        confidences, predictions = torch.max(probs, dim=-1)
        
    conf_list = confidences.tolist()
    pred_list = predictions.tolist()

    final_labels = ["O"] * len(words)
    final_confs = [0.0] * len(words)
    last_word_idx = None
    
    # Regex for currency-like amounts (e.g., 10.00 or 1,50)
    amount_pattern = re.compile(r'\d+[.,]\d{2}$')
    # Keywords that often cause false positives for the "TOTAL" label
    exclude_list = ["subtotal", "cash", "tax", "gst", "vat", "change", "items", "rounding", "balance"]

    for i, word_idx in enumerate(word_ids):
        # Only process the first token of each word
        if word_idx is not None and word_idx != last_word_idx:
            word_text = words[word_idx].strip()
            word_lower = word_text.lower()
            label = id2label[pred_list[i]]
            conf = conf_list[i]
            
            # --- Heuristic Filtering ---
            
            # Filter noise for short text in specific categories
            if "COMPANY" in label.upper() and len(word_text) < 2:
                label = "O"

            if "DATE" in label.upper() and len(word_text) < 2:
                label = "O"

            # Strict filtering for TOTAL amounts
            if "TOTAL" in label.upper():
                if any(x in word_lower for x in exclude_list):
                    label = "O"
                elif not any(c.isdigit() for c in word_text) and conf < 0.99:
                    label = "O"
                elif len(word_text) < 2 and (not word_text.isdigit() or conf < 0.95):
                    label = "O"

            if len(word_text) < 2 and conf < 0.95:
                label = "O"
            
            # Apply confidence threshold
            if conf >= threshold:
                final_labels[word_idx] = label
                final_confs[word_idx] = conf
            else:
                final_labels[word_idx] = "O"
                
            last_word_idx = word_idx

    # --- Global Selection Logic for TOTAL ---
    # Since a receipt usually has only one final total, we pick the best candidate
    total_indices = [i for i, l in enumerate(final_labels) if "TOTAL" in l.upper()]
    
    if total_indices:
        scored_candidates = []
        for idx in total_indices:
            box = boxes[idx]
            txt = words[idx].strip()
            conf = final_confs[idx]
            
            # Heuristic: Totals are usually at the bottom (y_norm) and right (x_norm)
            y_norm = box[3] / h  
            x_norm = box[2] / w  
            
            is_amount = 1.0 if amount_pattern.search(txt) else 0.0
            has_digits = 0.5 if any(c.isdigit() for c in txt) else 0.0
            
            # Calculate a weighted score to find the most likely Grand Total
            score = (y_norm * 1.5) + (x_norm * 0.5) + (is_amount * 2.0) + has_digits + conf
            scored_candidates.append((score, idx))
        
        # Select the candidate with the highest score
        best_idx = max(scored_candidates, key=lambda x: x[0])[1]
        
        # Set all other 'TOTAL' predictions back to 'O'
        for idx in total_indices:
            if idx != best_idx:
                final_labels[idx] = "O"

    return words, boxes, final_labels

def visualize_prediction(image_path, words, boxes, labels):
    """
    Overlays predicted bounding boxes and labels onto the image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Color scheme for different entities
    category_colors = {
        "TOTAL": (255, 0, 0),    # Red
        "DATE": (0, 255, 0),     # Green
        "ADDRESS": (0, 0, 255),  # Blue
        "COMPANY": (255, 255, 0),# Cyan/Yellow
        "O": (200, 200, 200)     # Light Grey
    }

    for box, label in zip(boxes, labels):
        # Extract base category from BIO format (e.g., B-TOTAL -> TOTAL)
        category = label.split("-")[-1] if "-" in label else label
        
        if category == "O":
            thickness = 1
            color = category_colors["O"]
        else:
            thickness = 2
            color = category_colors.get(category, (0, 0, 0))
            
            # Only put text for the start of an entity
            if label.startswith("B-"):
                cv2.putText(img, category, (box[0], box[1]-7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw the bounding box
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)

    # Display the result
    plt.figure(figsize=(10, 14))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Inference: {Path(image_path).name}")
    plt.show()


    from transformers import LayoutLMv3ForTokenClassification, AutoProcessor



# --- Model Loading ---

# Load the fine-tuned model and processor from the local directory
model_path = "layoutlmv3_sroie_final" 
trained_model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
trained_processor = AutoProcessor.from_pretrained(model_path)

# --- Test Data Paths ---

# Specify the paths to a sample image and its corresponding OCR box file
test_img = "/kaggle/input/datasets/maxbegal/dataset-layoutlm/data/test/img/X510056849111.jpg"
test_box = "/kaggle/input/datasets/maxbegal/dataset-layoutlm/data/test/box/X510056849111.txt"

# --- Inference and Visualization ---

# Execute the inference function with a confidence threshold (default 0.85)
words, boxes, labels = run_inference(
    test_img, 
    test_box, 
    trained_model, 
    trained_processor, 
    label2id
)

# Display the image with highlighted bounding boxes and entity labels
visualize_prediction(test_img, words, boxes, labels)

# Summary of identified entities
print(f"Results for: {test_img}")
for word, label in zip(words, labels):
    if label != "O":
        print(f"[{label}]: {word}")