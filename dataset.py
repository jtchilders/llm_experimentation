import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
from wikidata import WikiDataProcessor
from tokens import load_or_train_tokenizer
import logging
logger = logging.getLogger(__name__)
import time


class WikiDataset(Dataset):
   def __init__(self, data_file, tokens_file, context_size=5):
      self.tokenizer = load_or_train_tokenizer(data_file, tokens_file)
      self.data = []
      self.context_size = context_size

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
      return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# # Usage
# data_file = 'path_to_your_wikipedia_dataset.txt'
# tokens_file = 'path_to_tokenizer.json'
# dataset = WikiDataset(data_file, tokens_file, num_workers=8, chunk_size=100)
