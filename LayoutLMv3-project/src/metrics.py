import torch
import numpy as np
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def evaluate_model(pl_model, dataloader, label2id):
    """
    Evaluates the LayoutLMv3 model performance on a given dataloader.
    
    Args:
        pl_model: The trained LayoutLMv3Module (LightningModule).
        dataloader: PyTorch DataLoader (usually test_dataloader).
        label2id: Mapping of labels to integer IDs.
    """
    id2label = {v: k for k, v in label2id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract the underlying Transformers model and move to device
    model = pl_model.model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []

    print(f"🧐 Starting evaluation on {len(dataloader.dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move inputs to device, exclude labels for the forward pass
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].cpu().numpy() 

            outputs = model(**inputs)
            
            # Get predictions by taking the argmax of the logits
            predictions = outputs.logits.argmax(-1).detach().cpu().numpy()

            for i in range(len(labels)):
                # Mask out special tokens and padding (labeled as -100)
                mask = labels[i] != -100 
                
                # Convert numeric IDs back to string labels for seqeval
                true_seq = [id2label.get(l, "O") for l in labels[i][mask]]
                pred_seq = [id2label.get(p, "O") for p in predictions[i][mask]]
                
                if len(true_seq) > 0:
                    true_labels.append(true_seq)
                    pred_labels.append(pred_seq)

    print("\n" + "="*50)
    print("📊 TESTING RESULTS (SROIE):")
    print("="*50)
    
    if not true_labels:
        print("❌ Error: No labels were collected. Check the -100 mask logic.")
        return

    # Calculate overall metrics
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    print(f"✅ F1-Score:  {f1:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print("\n" + "-"*50)
    
    # Detailed report per entity category
    print(classification_report(true_labels, pred_labels))

# Prepare the data module splits
dm.setup() 

# Get the test dataloader
test_loader = dm.test_dataloader()

# Run the evaluation
evaluate_model(model, test_loader, label2id)