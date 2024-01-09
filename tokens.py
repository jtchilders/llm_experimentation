from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from wikidata import WikiDataProcessor
import logging
logger = logging.getLogger(__name__)

def train_tokenizer(data_file, tokens_file):
   # Initialize a tokenizer with BPE model
   tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
   tokenizer.pre_tokenizer = Whitespace()

   # Create a trainer for BPE
   trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

   # Process the data and train
   processor = WikiDataProcessor(data_file)
   tokenizer.train_from_iterator(processor.read_word_by_word(), trainer)

   # Save the tokenizer
   tokenizer.save(tokens_file)
   logging.info(f"Tokenizer trained and saved to '{tokens_file}'.")
   return tokenizer

def load_or_train_tokenizer(data_file, tokens_file):
   if os.path.exists(tokens_file):
      logging.info(f"Loading tokenizer from '{tokens_file}'.")
      return Tokenizer.from_file(tokens_file)
   else:
      logging.info(f"Token file '{tokens_file}' not found. Starting training.")
      return train_tokenizer(data_file, tokens_file)

# Example usage
# data_file = 'path_to_your_wikipedia_dataset.txt'
# tokens_file = 'path_to_tokenizer.json'

# tokenizer = load_or_train_tokenizer(data_file, tokens_file)
