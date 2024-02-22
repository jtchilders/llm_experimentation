import logging
import argparse
import os
import h5py
from wikidata import WikiDataProcessor
from tokens import load_or_train_tokenizer
import numpy as np

logger = logging.getLogger(__name__)

# takes a path to the input data and writes a tokenized version
def main():

   parser = argparse.ArgumentParser(description='Process data into a tokenized form.')
   parser.add_argument("-d","--data-file", help="input data file, should be text.", required=True)
   parser.add_argument("-t","--tokens-file", help="output tokens file", required=True)
   parser.add_argument("-o","--output-file", help="output tokeinzed data file", required=True)
   parser.add_argument("-c","--context-size", help="context size for training embeddings", type=int, default=2)
   parser.add_argument("-s","--context-symmetric", help="a symmetric context (before and after)", action="store_true")

   
   args = parser.parse_args()


   tokenize_data(args.data_file,args.tokens_file,args.output_file,args.context_size,args.context_symmetric)



def tokenize_data(data_file,tokens_file,output_file,context_size,context_symmetric):

   post_context_size = context_size if context_symmetric else 0
   tokenizer = load_or_train_tokenizer(data_file, tokens_file)
   vocab_size = tokenizer.get_vocab_size()

   logger.info(f"Reading {data_file}...")
   processor = WikiDataProcessor(data_file)
   
   contexts = []
   targets = []

   logger.info("Tokenizing data...")
   for sentence in processor.read_sentence_by_sentence():
      tokenized_sentence = tokenizer.encode(sentence).ids
      for i in range(context_size, len(tokenized_sentence) - post_context_size):
         # Include context before and after the target
         if context_symmetric:
            context_before = tokenized_sentence[i - context_size:i]
            context_after = tokenized_sentence[i + 1:i + 1 + context_size]
            context = context_before + context_after  # Combine before and after contexts
         else:
            context = tokenized_sentence[i - context_size:i]
         target = tokenized_sentence[i]
         contexts.append(context)
         targets.append(target)
   logger.info("Tokenizing done...")

   # Convert lists to numpy arrays for efficient storage
   contexts_array = np.array(contexts, dtype=np.int64)
   targets_array = np.array(targets, dtype=np.int64)

   # Save the tokenized data to an HDF5 file
   with h5py.File(output_file, 'w') as hf:
      hf.create_dataset("contexts", data=contexts_array)
      hf.create_dataset("targets", data=targets_array)

      # Add metadata as attributes
      hf.attrs['input_data_file'] = data_file
      hf.attrs['input_token_file'] = tokens_file
      hf.attrs['context_size'] = context_size
      hf.attrs['context_symmetric'] = context_symmetric

if __name__ == "__main__":
   main()