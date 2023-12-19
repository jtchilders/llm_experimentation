import os,glob,multiprocessing,time,datetime,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR,CosineAnnealingLR,StepLR
from torch.nn.functional import cosine_similarity
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from matplotlib import pyplot as plt
from functools import partial
import pickle

# Parameters
OUTPUT_DIR="results"
RUN_OUTPUT_DIR=os.path.join(OUTPUT_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
print(f'Output directory: {RUN_OUTPUT_DIR}')
token_fn = os.path.join(OUTPUT_DIR, "tokenizer-example.json")
CONTEXT_SIZE = 3
EMBEDDING_DIM = 250
EPOCHS = 5000
BATCH_SIZE = 400
WORD_CHUNK_SIZE = 3000
NUM_WORKERS = multiprocessing.cpu_count()
PRINT_EVERY = 5000
SAVE_INTERVAL = 100
PLOT_EVERY = 10

CHECKPOINT_DIR = os.path.join(RUN_OUTPUT_DIR, "checkpoints")
if len(sys.argv) > 1:
   CHECKPOINT_DIR = sys.argv[1]
LOG_DIR = os.path.join(RUN_OUTPUT_DIR, "tensorboard")
files_list = [f"/projects/datascience/parton/data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
files_dict = {split: f"/projects/datascience/parton/data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]}


# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



similar_words_source = [
   ('she', 'he'),
   ('rug', 'mat'),
   ('girl', 'boy'),
   ('train', 'car'),
]



# Tokenizer training
def train_tokenizer(files_list, token_fn):
   tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
   trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
   tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
   tokenizer.train(files_list, trainer)
   tokenizer.save(token_fn)

# Load or train tokenizer
if os.path.exists(token_fn):
   tokenizer = Tokenizer.from_file(token_fn)
else:
   train_tokenizer(files_list, token_fn)
   tokenizer = Tokenizer.from_file(token_fn)

similar_words = []
for word1, word2 in similar_words_source:
   if word1 in tokenizer.get_vocab() and word2 in tokenizer.get_vocab():
      similar_words.append((word1, word2))

def process_chunk(tokenizer, context_size, chunk):
   data = []
   tokenized_output = tokenizer.encode_batch(chunk)
   input_ids = [encoding.ids for encoding in tokenized_output]
   for ids in input_ids:
      for i in range(len(ids) - context_size):
         context = ids[i:i + context_size]
         target = ids[i + context_size]
         data.append((context, target))
   return data

def read_and_process_file(file_path, tokenizer, context_size, chunk_size=1000, num_workers=None):
   with open(file_path, 'r', encoding='utf-8') as file:
      chunks = [line.strip() for line in file]
   
   # Split chunks for multiprocessing
   chunks = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]

   # Process chunks in parallel
   with multiprocessing.Pool(processes=num_workers) as pool:
      results = pool.map(partial(process_chunk, tokenizer, context_size), chunks)

   # Flatten list of results
   data = [item for sublist in results for item in sublist]
   return data

def save_tokenized_data(data, file_path):
   with open(file_path, 'wb') as f:
      pickle.dump(data, f)

def load_tokenized_data(file_path):
   with open(file_path, 'rb') as f:
      return pickle.load(f)

# Custom Dataset
class TextDataset(Dataset):
   def __init__(self, file_path, tokenizer, context_size, chunk_size=1000, num_workers=None):
      tokenized_data_path = file_path + ".tokenized"
      if os.path.exists(tokenized_data_path):
         print("Loading pre-tokenized data...")
         self.data = load_tokenized_data(tokenized_data_path)
      else:
         print("Tokenizing data...")
         self.data = read_and_process_file(file_path, tokenizer, context_size, chunk_size, num_workers)
         print("Saving tokenized data for future use...")
         save_tokenized_data(self.data, tokenized_data_path)

   def __len__(self):
      return len(self.data)

   def __getitem__(self, idx):
      context, target = self.data[idx]
      return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# NGram Language Model
class NGramLanguageModeler(nn.Module):
   def __init__(self, vocab_size, embedding_dim, context_size):
      super(NGramLanguageModeler, self).__init__()
      self.embedding_dim = embedding_dim
      self.context_size = context_size
      self.embeddings = nn.Embedding(vocab_size, embedding_dim)
      self.linear1 = nn.Linear(context_size * embedding_dim, 128)
      self.linear2 = nn.Linear(128, vocab_size)

   def forward(self, inputs):
      embeds = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))
      out = F.relu(self.linear1(embeds))
      out = self.linear2(out)
      log_probs = F.log_softmax(out, dim=1)
      return log_probs


# Save model
def save_model(model, optimizer, epoch, step, checkpoint_dir):
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:06}_step_{step:06}.pth')
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def load_latest_checkpoint(checkpoint_dir, model):
   checkpoints = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pth')]
   latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
   print(f"Loading checkpoint: {latest_checkpoint}")

   checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))

   # Check if the model is wrapped in DataParallel
   is_model_data_parallel = isinstance(model, nn.DataParallel)

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




def plot_embeddings_with_cosine_similarity(model, tokenizer, epoch, similar_words, plot_dir='plots'):
   # Ensure the plotting directory exists
   os.makedirs(plot_dir, exist_ok=True)

   # Extract embeddings
   embeddings = model.embeddings.weight.detach().cpu()
   vocab = tokenizer.get_vocab()

   # Plotting the embeddings for specified similar words
   plt.figure(figsize=(10, 8))
   for word1, word2 in similar_words:
      if word1 in vocab and word2 in vocab:
         index1 = vocab[word1]
         index2 = vocab[word2]
         word1_embedding = embeddings[index1].unsqueeze(0)  # Add batch dimension
         word2_embedding = embeddings[index2].unsqueeze(0)  # Add batch dimension
         similarity = cosine_similarity(word1_embedding, word2_embedding).item()
         print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")

         # Plot the embeddings
         plt.scatter(embeddings[index1, 0], embeddings[index1, 1], label=f"{word1} ({similarity:.2f})")
         plt.scatter(embeddings[index2, 0], embeddings[index2, 1], label=f"{word2} ({similarity:.2f})")
         plt.text(embeddings[index1, 0], embeddings[index1, 1], word1, fontsize=9)
         plt.text(embeddings[index2, 0], embeddings[index2, 1], word2, fontsize=9)

   plt.xlabel('Dimension 1')
   plt.ylabel('Dimension 2')
   plt.title(f'Word Embeddings and Cosine Similarities at Epoch {epoch+1}')
   plt.legend()
   plt.savefig(os.path.join(plot_dir, f'embeddings_cosine_similarity_epoch_{epoch+1}.png'))
   plt.savefig(os.path.join(plot_dir, f'embeddings_cosine_similarity_epoch_{epoch+1}.pdf'))
   plt.close()




def compute_distance(embeddings, word_index_1, word_index_2):
    """Computes the Euclidean distance between two word embeddings."""
    vec1 = embeddings[word_index_1]
    vec2 = embeddings[word_index_2]
    return torch.dist(vec1, vec2).item()

def compute_cosine_similarity(embeddings, word_index_1, word_index_2):
    """Computes the cosine similarity between two word embeddings."""
    vec1 = embeddings[word_index_1].unsqueeze(0)  # Add batch dimension
    vec2 = embeddings[word_index_2].unsqueeze(0)  # Add batch dimension
    return cosine_similarity(vec1, vec2).item()

# Training loop
def train(model, data_loader, optimizer, criterion, scheduler, epochs, print_every, save_interval, checkpoint_dir, log_dir, starting_epoch=0):
   writer = SummaryWriter(log_dir)
   model.train()
   for epoch in range(starting_epoch,starting_epoch+epochs):
      total_loss = 0.0
      running_loss = 0.0
      running_time = time.time()
      
      for step, (context, target) in enumerate(data_loader):
         context, target = context.to(device), target.to(device)
         optimizer.zero_grad()
         log_probs = model(context)
         loss = criterion(log_probs, target)
         loss.backward()
         optimizer.step()
         scheduler.step()
         total_loss += loss.item()
         running_loss += loss.item()

         if (step + 1) % print_every == 0:
            avg_running_loss = running_loss / print_every
            avg_rate =  print_every * BATCH_SIZE / (time.time() - running_time)
            print(f"Epoch [{epoch+1}/{epochs+starting_epoch}], Step [{step+1}/{len(data_loader)}], Loss: {running_loss:.4e}, Avg Rate: {avg_rate:.4e}")
            writer.add_scalar('Metrics/Running Average Loss', running_loss, epoch * len(data_loader) + step)
            writer.add_scalar('Metrics/Running Average Rate', avg_rate, epoch * len(data_loader) + step)
            # Log the current learning rate
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Metrics/Learning Rate', current_lr, epoch * len(data_loader) + step)

            if isinstance(model, nn.DataParallel):
               embeddings = model.module.embeddings.weight.data
            else:
               embeddings = model.embeddings.weight.data
            if len(similar_words) == 0 and epoch == 0:
               print('no similar words.')
            for word1, word2 in similar_words:
               index1 = tokenizer.token_to_id(word1)
               index2 = tokenizer.token_to_id(word2)
               similarity = compute_cosine_similarity(embeddings, index1, index2)
               writer.add_scalar(f'Cosine Similarity/{word1}-{word2}', similarity, epoch * len(data_loader) + step)
            running_loss = 0.0  # Reset running loss after logging
            running_time = time.time()
            sys.stdout.flush()
            sys.stderr.flush()
      
      if isinstance(model, nn.DataParallel):
         embeddings = model.module.embeddings.weight.data
      else:
         embeddings = model.embeddings.weight.data
      for word1, word2 in similar_words:
         index1 = tokenizer.token_to_id(word1)
         index2 = tokenizer.token_to_id(word2)
         similarity = compute_cosine_similarity(embeddings, index1, index2)
         writer.add_scalar(f'Cosine Similarity/{word1}-{word2}', similarity,  (epoch+1) * len(data_loader))
      save_model(model, optimizer, epoch+1,0, checkpoint_dir)

      avg_loss = total_loss / len(data_loader)
      writer.add_scalar('Metrics/Average Loss', avg_loss, epoch)
      # Log the current learning rate
      current_lr = scheduler.get_last_lr()[0]
      writer.add_scalar('Metrics/Learning Rate', current_lr, (epoch+1) * len(data_loader))
      print(f"Epoch [{epoch+1}/{epochs+starting_epoch}] completed, Total Loss: {avg_loss:.4e}")

   writer.close()

# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.get_vocab_size()

model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
if torch.cuda.device_count() > 1:
   print(f"Using {torch.cuda.device_count()} GPUs!")
   model = nn.DataParallel(model)
model.to(device)

checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_*.pth')))
starting_epoch = 0
starting_step = 0
if len(checkpoints) > 0:
   checkpoint = load_latest_checkpoint(CHECKPOINT_DIR,model)
   starting_epoch = checkpoint['epoch']
   starting_step = checkpoint['step']
   model.load_state_dict(checkpoint['model_state_dict'])
   model.to(device)

dataset = TextDataset(files_dict['train'], tokenizer, CONTEXT_SIZE, chunk_size=WORD_CHUNK_SIZE, num_workers=NUM_WORKERS)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=0.05)
# scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, step_size_up=2000000, mode='triangular2')
scheduler = StepLR(optimizer, step_size=10e6, gamma=0.7)
criterion = nn.NLLLoss()

train(model, data_loader, optimizer, criterion, scheduler, EPOCHS, PRINT_EVERY, SAVE_INTERVAL, CHECKPOINT_DIR, LOG_DIR, starting_epoch)
