import torch
import os
import json
from config import Config
from dataset import get_dataloader, EmotionDataset
from torch.utils.data import DataLoader
from model import GRUAttention

def run_final_evaluation(task_name):
    """
    Performs a final evaluation on the test set and exports a performance report.
    """
    config = Config(task_name)
    print(f"\n{'='*60}")
    print(f"📊 Preparing Evaluation for Task: {config.task_name.upper()}")
    
    # Verify if the trained weights exist
    if not os.path.exists(config.model_save_path):
        print(f"❌ Error: Model weights not found at {config.model_save_path}")
        print(f"Please run 'python train.py' for this task first. Skipping...\n")
        return None

    # 1. Reconstruct Vocabulary for ID Alignment
    # We use the training CSV to ensure the word-to-id mapping matches the trained model.
    _, _, vocab = get_dataloader(
        train_csv=config.train_csv, 
        val_csv=config.val_csv, 
        config=config
    )
    
    # 2. Load the Unseen Test Dataset
    test_dataset = EmotionDataset(config.test_csv, vocab, config.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 3. Model Initialization & Weight Loading
    model = GRUAttention(config).to(config.device)
    # Using weights_only=True for safe unpickling as per PyTorch best practices
    model.load_state_dict(torch.load(config.model_save_path, map_location=config.device, weights_only=True))
    model.eval() 
    
    print(f"🚀 Evaluation in progress for {config.task_name.upper()}...")
    correct = 0
    total = 0
    
    # 4. Inference Phase (Gradient calculation disabled for efficiency)
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
            
            # Unpack outputs; attention weights are ignored during batch testing
            outputs, _ = model(batch_x)
            
            # Determine predicted class (Argmax)
            _, predicted = torch.max(outputs.data, 1)
            
            # Aggregate statistics
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    # 5. Result Calculation
    accuracy = 100 * correct / total
    print(f"🏆 Final Test Accuracy: {accuracy:.2f}%")
    
    # 6. Export Report for Streamlit Integration
    # This JSON allows your Demo to show a summary table of all models.
    report = {
        "task": task_name,
        "test_accuracy": round(accuracy, 2),
        "total_samples": total,
        "model_architecture": "Bi-GRU + Attention",
        "vocab_size": config.vocab_size,
        "status": "Verified"
    }
    
    # 1. Get the current directory where test.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(current_dir, 'report')

    report_path = os.path.join(report_dir, f"{task_name}_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"📄 Report exported to: {report_path}")
    print(f"{'='*60}")
    
    return report

if __name__ == "__main__":
    # Batch processing pipeline for multi-task evaluation
    tasks_to_evaluate = ['emotion', 'sst2', 'imdb']
    all_reports = []
    
    print("🌟 Initializing Automated Multi-Task Evaluation Pipeline...")
    
    for task in tasks_to_evaluate:
        res = run_final_evaluation(task)
        if res:
            all_reports.append(res)
            
    print("\n✅ All evaluations complete.")
    print(f"Total tasks verified: {len(all_reports)}/{len(tasks_to_evaluate)}")