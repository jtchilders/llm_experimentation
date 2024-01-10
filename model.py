import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import logging
logger = logging.getLogger(__name__)


class NGramLanguageModeler(nn.Module):
   def __init__(self, vocab_size, embedding_dim, context_size):
      super(NGramLanguageModeler, self).__init__()
      self.embedding_dim = embedding_dim
      self.context_size = context_size
      self.embeddings = nn.Embedding(vocab_size, embedding_dim)
      self.linear1 = nn.Linear(context_size * embedding_dim, 128)
      self.linear2 = nn.Linear(128, vocab_size)

   def forward(self, inputs):
      embeds = self.embeddings(inputs)
      embeds = embeds.view((-1, self.context_size * self.embedding_dim))
      out = F.relu(self.linear1(embeds))
      out = self.linear2(out)
      log_probs = F.log_softmax(out, dim=1)
      return log_probs


def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
   model_to_save = model.module if isinstance(model, nn.DataParallel) else model
   checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:06}_step_{step:06}.pth')
   torch.save({
      'epoch': epoch,
      'step': step,
      'model_state_dict': model_to_save.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
   }, checkpoint_path)

def load_checkpoint(checkpoint_path, model):
   checkpoint = torch.load(checkpoint_path)

   # Check if the model is wrapped in DataParallel
   is_model_data_parallel = isinstance(model, DDP)

   # Prepare new state dict with appropriate key formatting
   new_state_dict = {}
   for k, v in checkpoint['model_state_dict'].items():
      if is_model_data_parallel:
         # Add 'module.' prefix if missing
         new_k = f'module.{k}' if not k.startswith('module.') else k
      else:
         # Remove 'module.' prefix if present
         new_k = k.replace('module.', '') if k.startswith('module.') else k
      new_state_dict[new_k] = v

   # Load the adjusted state dict
   checkpoint['model_state_dict'] = new_state_dict

   return checkpoint

def count_parameters(model):
   return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
# vocab_size = 10000  # replace with your actual vocab size
# embedding_dim = 200  # you can choose an appropriate embedding dimension

# model = WordEmbeddingModel(vocab_size, embedding_dim)
