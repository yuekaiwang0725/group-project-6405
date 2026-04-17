import os
import torch

class Config:
    def __init__(self, task_name='emotion'):
        self.task_name = task_name
        
        # --- Automatic path location logic ---
        # 1. Get the absolute path of this config.py file
        # Result example: C:/Users/xxx/Desktop/group-project-6405/gru_attention
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up one level to find the project root directory
        # Result example: C:/Users/xxx/Desktop/group-project-6405
        root_dir = os.path.dirname(current_dir)
        
        # 3. Construct paths for data folder and model save folder
        # This ensures the program can automatically compute paths regardless of where your project folder is placed
        data_base_path = os.path.join(root_dir, 'data/processed')
        self.save_dir = os.path.join(current_dir, 'trained_models')

        # ==========================================
        # Automatically switch to task-specific parameters based on different tasks (using dynamically constructed paths)
        # ==========================================
        if task_name == 'emotion':
            self.num_classes = 6
            self.seq_len = 50
            # Use os.path.join to automatically handle path slash differences between Windows/Linux
            self.train_csv = os.path.join(data_base_path, 'emotion_train.csv')
            self.val_csv = os.path.join(data_base_path, 'emotion_val.csv')
            self.test_csv = os.path.join(data_base_path, 'emotion_test.csv')
            self.model_save_path = os.path.join(self.save_dir, 'gru_emotion_model.pth')
            
        elif task_name == 'imdb':
            self.num_classes = 2
            self.seq_len = 256
            self.train_csv = os.path.join(data_base_path, 'imdb_train.csv')
            self.val_csv = os.path.join(data_base_path, 'imdb_val.csv')
            self.test_csv = os.path.join(data_base_path, 'imdb_test.csv')
            self.model_save_path = os.path.join(self.save_dir, 'gru_imdb_model.pth')
            
        elif task_name == 'sst2':
            self.num_classes = 2
            self.seq_len = 50
            self.train_csv = os.path.join(data_base_path, 'sst2_train.csv')
            self.val_csv = os.path.join(data_base_path, 'sst2_val.csv')
            self.test_csv = os.path.join(data_base_path, 'sst2_test.csv')
            self.model_save_path = os.path.join(self.save_dir, 'gru_sst2_model.pth')

        # Common parameters remain unchanged...
        self.vocab_size = 8000     
        self.embedding_dim = 128   
        self.hidden_dim = 256      
        self.batch_size = 64       
        self.learning_rate = 1e-3  
        self.num_epochs = 20       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')