import logging
import argparse
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore warnings
from time import gmtime, strftime
from wikidata import WikiDataProcessor
from tokens import load_or_train_tokenizer
from dataset import WikiDataset
from model import NGramLanguageModeler
from trainer import train
import torch
from mpi import COMM_WORLD,rank,world_size,local_rank
logger = logging.getLogger(__name__)

def main():
   # Set up command line argument parser
   parser = argparse.ArgumentParser(description='Process some data.')
   parser.add_argument('-v', '--verbose', action='count', default=0,
                     help='increase output verbosity')
   parser.add_argument('-d','--device', type=str, default='cuda',
                        help='device to use for training (cuda or cpu)')
   parser.add_argument('-o','--output-dir', type=str, default=os.path.join('results', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
   parser.add_argument('-b','--batch-size', type=int, default=256)
   parser.add_argument('-e','--epochs', type=int, default=10)
   parser.add_argument('-c','--context-size', type=int, default=2)
   parser.add_argument('-l','--learning-rate', type=float, default=0.001)
   parser.add_argument('-i','--log-interval', type=int, default=500)
   parser.add_argument('-m','--embedding-dim', type=int, default=200)
   args = parser.parse_args()


   # Set up logging
   log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
   log_level = logging.WARN
   if args.verbose > 0:
      log_level = logging.DEBUG
   elif rank == 0:
      log_level = logging.INFO
   logging.basicConfig(level=log_level,
                       format=log_format,
                       datefmt='%Y-%m-%d %H:%M:%S')

   OUTPUT_DIR=args.output_dir
   logger.info("Output directory: " + OUTPUT_DIR)

   epochs = args.epochs
   lr = args.learning_rate
   batch_size = args.batch_size
   log_interval = args.log_interval
   context_size = args.context_size

   data_file = '/lus/eagle/projects/atlas_aesp/data/wikitext-103-raw/wiki.train.raw'
   tokens_file = '/lus/eagle/projects/atlas_aesp/data/wikitext-103-raw/wiki.train.raw.json'
   embedding_dim = args.embedding_dim # you can choose an appropriate embedding dimension

   # Log debug output
   logger.debug(f"Running main function with rank {rank} and world size {world_size}")

   # Set device to cuda if available
   if 'cuda' in args.device:
      device = torch.device(f'cuda:{local_rank}')
   else:
      device = torch.device(args.device)
   logger.info(f"Using device: {device}")

   # tokenizer = load_or_train_tokenizer(data_file, tokens_file)
   dataset = WikiDataset(data_file, tokens_file,context_size)
   logger.info(f'vocab size: {dataset.tokenizer.get_vocab_size()}')
   model = NGramLanguageModeler(dataset.tokenizer.get_vocab_size(), embedding_dim, context_size).to(device)

   train(model, dataset, epochs, lr, batch_size, log_interval, device, OUTPUT_DIR)

# Call the main function
if __name__ == "__main__":
   main()