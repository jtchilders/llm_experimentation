import logging
import argparse
import os
import datetime
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore warnings
from time import gmtime, strftime
from wikidata import WikiDataProcessor
from tokens import load_or_train_tokenizer
from dataset import WikiDataset
from model import NGramLanguageModeler,count_parameters
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
   parser.add_argument('-o','--output-dir', help='output directory for checkpoints, tensorboard, and logs.', type=str, default=os.path.join('results', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
   parser.add_argument('-b','--batch-size', help='batch size', type=int, default=256)
   parser.add_argument('-e','--epochs', help='number of training epochs', type=int, default=10)
   parser.add_argument('-c','--context-size', help='context size for training embeddings', type=int, default=2)
   parser.add_argument('-l','--learning-rate', help='learning rate', type=float, default=0.001)
   parser.add_argument('-i','--log-interval', help='print every N steps during training', type=int, default=500)
   parser.add_argument('-m','--embedding-dim', help='embedding dimension', type=int, default=200)
   parser.add_argument('--pytorch-profiler', action='store_true', default=False,help='enable pytorch profiler')
   parser.add_argument('--profile-steps', help='number of steps to profile', type=int, default=1000)
   parser.add_argument('--train-file', help='training data file', type=str, default='/lus/eagle/projects/datascience/parton/data/wikitext-103-raw/wiki.train.raw')
   parser.add_argument('--use-synthetic', action='store_true', default=False, help='use synthetic data')
   parser.add_argument('--vocab-size', help='vocab size', type=int, default=30000)
   parser.add_argument('--synthetic-size', help='synthetic data size', type=int, default=10000000)
   parser.add_argument('--logdir-append', help='logdir append string', type=str, default='')
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

   OUTPUT_DIR=args.output_dir+args.logdir_append
   logger.info("Output directory: " + OUTPUT_DIR)

   epochs = args.epochs
   lr = args.learning_rate
   batch_size = args.batch_size
   log_interval = args.log_interval
   context_size = args.context_size

   train_file = args.train_file
   tokens_file = train_file + '.json'
   embedding_dim = args.embedding_dim # you can choose an appropriate embedding dimension
   tokenized_data_filename = train_file


   # write all config parameters to output direction in json format
   if rank == 0:
      os.makedirs(OUTPUT_DIR, exist_ok=True)
      config = {
         'epochs': epochs,
         'lr': lr,
         'batch_size': batch_size,
         'log_interval': log_interval,
         'context_size': context_size,
         'embedding_dim': embedding_dim,
         'train_file': train_file,
         'tokens_file': tokens_file,
         'tokenized_data_filename': tokenized_data_filename,
         'world_size': world_size,
         'pytorch_profiler': args.pytorch_profiler,
         'profile_steps': args.profile_steps,
         'use_synthetic': args.use_synthetic,
         'vocab_size': args.vocab_size,
         'synthetic_size': args.synthetic_size,
      }
      with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
         json.dump(config, f, indent=4, sort_keys=True)

   # Log debug output
   logger.debug(f"Running main function with rank {rank} and world size {world_size}")

   # Set device to cuda if available
   if 'cuda' in args.device:
      device = torch.device(f'cuda:{local_rank}')
   else:
      device = torch.device(args.device)
   logger.info(f"Using device: {device}")

   # tokenizer = load_or_train_tokenizer(data_file, tokens_file)
   dataset = WikiDataset(train_file, tokens_file, tokenized_data_filename=tokenized_data_filename, context_size=context_size,use_synthetic=args.use_synthetic,vocab_size=args.vocab_size,synthetic_size=args.synthetic_size)
   logger.info(f'vocab size: {dataset.vocab_size}')
   model = NGramLanguageModeler(dataset.vocab_size, embedding_dim, context_size).to(device)

   total_params = count_parameters(model)
   logger.info(f"Total number of model parameters: {total_params}")

  
   train(model, dataset, epochs, lr, batch_size, 
         log_interval, device, OUTPUT_DIR, 
         profile_steps=args.profile_steps, pytorch_profiler=args.pytorch_profiler)

# Call the main function
if __name__ == "__main__":
   main()
      