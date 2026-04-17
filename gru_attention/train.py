import torch
import torch.nn as nn
from config import Config
from dataset import get_dataloader
from model import GRUAttention  # Import the model you initially wrote

def train_single_task(task_name):
    # 1. Initialize dynamic configuration
    config = Config(task_name)
    print(f"\n{'='*50}")
    print(f"🚀 Starting task: {config.task_name.upper()}")
    print(f"🎯 Number of classes: {config.num_classes} | Truncation length: {config.seq_len}")
    print(f"Current computing device: {config.device}")
    print(f"{'='*50}")
    
    # 2. Dynamically load corresponding task data (paths automatically obtained from config)
    print(f"Building {task_name.upper()} vocabulary and loading data...")
    train_loader, val_loader, vocab = get_dataloader(
        train_csv=config.train_csv, 
        val_csv=config.val_csv, 
        config=config
    )
    
    # 3. Instantiate model and move to GPU/CPU
    model = GRUAttention(config).to(config.device)
    
    # 4. Define judge (loss function) and coach (optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 5. Start the long training loop
    print("====== Start training ======")
    for epoch in range(config.num_epochs):
        model.train() # Enable training mode
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            # Move data to GPU/CPU
            batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
            
            # Clear gradients from previous round
            optimizer.zero_grad()
            
            # Forward pass: model predicts emotion
            outputs, _ = model(batch_x)
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass: update GRU and Attention weights
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_acc = 100 * correct / total
        
        # --- After each epoch, test on validation set (val) ---
        model.eval() # Enable evaluation mode
        val_correct = 0
        val_total = 0
        with torch.no_grad(): # No cheating during test (no gradient computation)
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(config.device), val_y.to(config.device)
                val_outputs, _ = model(val_x)
                _, val_pred = torch.max(val_outputs.data, 1)
                val_total += val_y.size(0)
                val_correct += (val_pred == val_y).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}] | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        
    # --- After exiting epoch loop, save current task's model ---
    print(f"\n✅ {task_name.upper()} training completed! Saving model weights...")
    torch.save(model.state_dict(), config.model_save_path)
    print(f"💾 Model successfully saved as: {config.model_save_path}")

if __name__ == "__main__":
    # Automatic pipeline: define all tasks you need to run
    # IMDb sentences are long and time-consuming, run last
    tasks_to_run = ['emotion', 'sst2', 'imdb']
    
    print("🌟 Preparing to start fully automated batch training pipeline...")
    for task in tasks_to_run:
        train_single_task(task)
        
    print("\n🎉🎉🎉 Congratulations! All three datasets have been trained, and three independent model files have been generated!")