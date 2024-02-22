import torch
from torch.utils.data import Dataset,IterableDataset
from multiprocessing import Pool
from wikidata import WikiDataProcessor
from tokens import load_or_train_tokenizer
import h5py
import numpy as np
import logging
logger = logging.getLogger(__name__)
import time,os


class StreamingWikiDataset(IterableDataset):
   def __init__(self, h5_file):
      super(StreamingWikiDataset, self).__init__()
      self.h5_file = h5_file
      with h5py.File(self.h5_file, 'r') as f:
         self.context_size = f.attrs['context_size']
         self.input_data_file = f.attrs['input_data_file']
         self.input_token_file = f.attrs['input_token_file']
         self.context_symmetric = f.attrs['context_symmetric']
         self.length = f['contexts'].shape[0]
   
   def __len__(self):
      return self.length

   def __iter__(self):
      # Open the HDF5 file and yield one context-target pair at a time
      with h5py.File(self.h5_file, 'r') as file:
         contexts = file['contexts']
         targets = file['targets']
         for context, target in zip(contexts, targets):
            yield torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class DistributedStreamingWikiDataset(IterableDataset):
   def __init__(self, h5_file, rank=0, world_size=1):
      super(DistributedStreamingWikiDataset, self).__init__()
      self.h5_file = h5_file
      self.rank = rank
      self.world_size = world_size
      self.use_synthetic = False
      with h5py.File(self.h5_file, 'r') as f:
         self.context_size = f.attrs['context_size']
         self.input_data_file = f.attrs['input_data_file']
         self.input_token_file = f.attrs['input_token_file']
         self.context_symmetric = f.attrs['context_symmetric']
         self.tokenizer = load_or_train_tokenizer(self.input_data_file, self.input_token_file)
         self.length = f['contexts'].shape[0]

   def __len__(self):
      return int(self.length / self.world_size) + 1
   
   def __iter__(self):
      worker_info = torch.utils.data.get_worker_info()
      
      # Evenly split dataset among ranks in distributed training
      with h5py.File(self.h5_file, 'r') as f:
         per_rank = int(self.length / self.world_size)
         iter_start = self.rank * per_rank
         iter_end = iter_start + per_rank if self.rank != self.world_size - 1 else self.length

      return self._generate_data(iter_start, iter_end)

   def _generate_data(self, start, end):
      with h5py.File(self.h5_file, 'r') as f:
         contexts = f['contexts'][start:end]
         targets = f['targets'][start:end]
      for context, target in zip(contexts, targets):
         yield torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class WikiDataset(Dataset):
   def __init__(self, data_file, tokens_file, tokenized_data_filename='', use_synthetic=False, context_size=5, synthetic_size=1000, vocab_size=10000):
      self.use_synthetic = use_synthetic
      self.data = []
      self.context_size = context_size
      self.vocab_size = vocab_size

      if use_synthetic:
         # Generate synthetic data
         logger.info(f"Generating synthetic data with {synthetic_size} examples...")
         self.data = [(
            torch.randint(vocab_size, (2 * context_size,)).tolist(),  # Double context_size for before and after
            torch.randint(0, vocab_size, (1,)).item()
         ) for _ in range(synthetic_size)]
      else:
         self.tokenizer = load_or_train_tokenizer(data_file, tokens_file)
         self.vocab_size = self.tokenizer.get_vocab_size()
         logger.info(f"Reading {data_file}...")
         processor = WikiDataProcessor(data_file)
         logger.info("Tokenizing data...")
         for sentence in processor.read_sentence_by_sentence():
            tokenized_sentence = self.tokenizer.encode(sentence).ids
            for i in range(context_size, len(tokenized_sentence) - context_size):
               # Include context before and after the target
               context_before = tokenized_sentence[i - context_size:i]
               context_after = tokenized_sentence[i + 1:i + 1 + context_size]
               context = context_before + context_after  # Combine before and after contexts
               target = tokenized_sentence[i]
               self.data.append((context, target))
         logger.info("Tokenizing done...")

   def __len__(self):
      return len(self.data)

   def __getitem__(self, idx):
      context, target = self.data[idx]
      context, target = torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
      return context, target

# only includes context before the target token
class WikiDatasetB(Dataset):
   def __init__(self, data_file, tokens_file, tokenized_data_filename='', use_synthetic=False, context_size=5, synthetic_size=1000, vocab_size=10000):
      self.use_synthetic = use_synthetic
      self.data = []
      self.context_size = context_size
      self.vocab_size = vocab_size
      self.use_synthetic = use_synthetic

      if use_synthetic:
         # Generate synthetic data
         logger.info(f"Generating synthetic data with {synthetic_size} examples...")
         self.data = [(torch.randint(vocab_size, ( context_size,)).tolist(), torch.randint(0, vocab_size, (1,)).item()) for _ in range(synthetic_size)]
         context, target = self.data[0]
         self.tokenizer = None
      else:
         self.tokenizer = load_or_train_tokenizer(data_file, tokens_file)
         self.vocab_size = self.tokenizer.get_vocab_size()
         logger.info(f"Reading {data_file}...")
         processor = WikiDataProcessor(data_file)
         logger.info("Tokenizing data...")
         for sentence in processor.read_sentence_by_sentence():
            tokenized_sentence = self.tokenizer.encode(sentence).ids
            for i in range(context_size, len(tokenized_sentence) - context_size):
               context = tokenized_sentence[i - context_size:i]
               target = tokenized_sentence[i]
               self.data.append((context, target))
         logger.info("Tokenizing done...")

   def __len__(self):
      return len(self.data)

   def __getitem__(self, idx):
      context, target = self.data[idx]
      context, target = torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
      return context, target
