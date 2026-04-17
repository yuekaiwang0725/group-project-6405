import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class Vocab:
    def __init__(self, texts, max_size=5000):
        import re
        from collections import Counter
        counter = Counter()
        for text in texts:
            # Clean and split
            clean_words = self.clean_text(text).split()
            counter.update(clean_words)
        
        # Build word list
        self.vocab_words = ['<PAD>', '<UNK>'] + [word for word, count in counter.most_common(max_size - 2)]
        
        # 💡 FORCE INT KEYS: Ensure the keys are standard Python integers
        self.word2id = {str(word): int(idx) for idx, word in enumerate(self.vocab_words)}
        self.id2word = {int(idx): str(word) for idx, word in enumerate(self.vocab_words)}
        
    def clean_text(self, text):
        import re
        text = str(text).lower().replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def text_to_ids(self, text, seq_len):
        clean_words = self.clean_text(text).split()
        # Look up ID, default to 1 (<UNK>)
        ids = [self.word2id.get(word, 1) for word in clean_words]
        
        if len(ids) >= seq_len:
            return ids[:seq_len], seq_len
        else:
            return ids + [0] * (seq_len - len(ids)), len(ids)

    def ids_to_tokens(self, ids, num_used):
        # 💡 CRITICAL FIX: Ensure idx is treated as a clean integer
        actual_ids = ids[:num_used]
        tokens = []
        for idx in actual_ids:
            # Convert to standard int to match dictionary key
            lookup_key = int(idx) 
            token = self.id2word.get(lookup_key, f"ID:{lookup_key}") 
            tokens.append(token)
        return tokens

    def __len__(self):
        return len(self.vocab_words)
    
    
class EmotionDataset(Dataset):
    """
    PyTorch Dataset class: responsible for feeding data to the model
    """
    def __init__(self, csv_file, vocab, seq_len):
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Extract text and label columns
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        
        self.vocab = vocab
        self.seq_len = seq_len
        
    def __len__(self):
        # Tell PyTorch the size of the dataset
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Get the idx-th data sample
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Call vocab to convert text to numeric IDs
        text_ids, _ = self.vocab.text_to_ids(text, self.seq_len)
        
        # Return in Tensor format that PyTorch can compute
        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def get_dataloader(train_csv, val_csv, config):
    """
    Quick interface function to get DataLoader
    """
    # 1. First read training set texts to build a unified vocabulary
    train_df = pd.read_csv(train_csv)
    train_texts = train_df['text'].tolist()
    
    # 2. Build vocabulary (limit vocabulary size, e.g., only keep the most frequent 8000 words)
    vocab = Vocab(train_texts, max_size=config.vocab_size)
    print(f"Vocabulary built! Contains {len(vocab)} words total.")
    
    # 3. Instantiate datasets
    train_dataset = EmotionDataset(train_csv, vocab, config.seq_len)
    val_dataset = EmotionDataset(val_csv, vocab, config.seq_len)
    
    # 4. Wrap into DataLoader that outputs in batches
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab