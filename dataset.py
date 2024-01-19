import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
from wikidata import WikiDataProcessor
from tokens import load_or_train_tokenizer
import h5py
import numpy as np
import logging
logger = logging.getLogger(__name__)
import time,os


class WikiDataset(Dataset):
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
               target = tokenized_sentence[i-1]
               self.data.append((context, target))
         logger.info("Tokenizing done...")

   def __len__(self):
      return len(self.data)

   def __getitem__(self, idx):
      context, target = self.data[idx]
      context, target = torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
      return context, target
