import os,glob
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
import pickle

# Parameters
token_fn = "tokenizer-example.json"
CONTEXT_SIZE = 3
EMBEDDING_DIM = 50
EPOCHS = 5000
BATCH_SIZE = 32
PRINT_EVERY = 100
SAVE_INTERVAL = 100
PLOT_EVERY = 10
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs/model_training"

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# List of sentences
sentences = [
   "The cat sits on the mat.",
   "A dog lies on the rug.",
   "She reads a book in the park.",
   "The dog plays in the park",
   "She reads on a mat.",
   "He reads a book on a mat.",
   "A dog plays in the park",
   "He plays in the park",
   "She plays in the park",
   "A cat lies on the mat.",
   "He reads in the park.",
   "The cat sits on the mat.",
   "A dog lies on the rug.",
   "She reads a book in the park.",
   "He plays the guitar for fun.",
   "Birds fly across the clear sky.",
   "Fish swim in the deep sea.",
   "The sun rises in the east.",
   "Stars twinkle in the night sky.",
   "Children play games at school.",
   "She paints a picture with colors.",
   "He drives a car to work.",
   "They watched a movie last night.",
   "The chef cooks tasty meals.",
   "Farmers grow crops in the fields.",
   "Artists create beautiful artworks.",
   "Carpenters build houses and furniture.",
   "Teachers educate students in schools.",
   "Doctors treat patients in hospitals.",
   "Athletes run races in competitions.",
   "Musicians perform concerts for audiences.",
   "A baker bakes bread and cakes.",
   "Gardeners plant flowers and trees.",
   "Writers publish books and articles.",
   "Photographers take photos of nature.",
   "A tailor sews clothes and dresses.",
   "Chefs prepare delicious dishes.",
   "Joggers run in the park every morning.",
   "The moon glows in the dark sky.",
   "The river flows into the ocean.",
   "Trains travel across the country.",
   "She sings songs beautifully.",
   "He plays piano in a band.",
   "The horse gallops across the field.",
   "A bird builds a nest in a tree.",
   "The clock ticks every second.",
   "She writes poems about nature.",
   "A fisherman catches fish in the lake.",
   "A baby sleeps peacefully in the crib.",
   "A student studies for exams.",
   "The wind blows the leaves around.",
   "A painter decorates homes.",
   "He fixes cars in the garage.",
   "A dancer performs on stage.",
   "A cat chases a mouse.",
   "The teacher explains the lesson.",
   "She cuts the cake into slices.",
   "A mechanic repairs machines.",
   "A pilot flies an airplane.",
   "He swims laps in the pool.",
   "She jogs every day for health.",
   "A squirrel gathers nuts for winter.",
   "Rain falls on the rooftop.",
   "A policeman directs traffic.",
   "The scientist conducts experiments.",
   "She knits a warm sweater.",
   "A waiter serves food to customers.",
   "He plays soccer on weekends.",
   "A nurse cares for the sick.",
   "She types a report on the computer.",
   "A cat purrs when happy.",
   "Children laugh at the clown.",
   "He ordered pizza for dinner.",
   "A teacher marks the tests.",
   "They celebrate birthdays with joy.",
   "He reads newspapers every day.",
   "A singer rehearses for the show.",
   "She dances with grace.",
   "A postman delivers letters and parcels.",
   "The moonlight reflects on the lake.",
   "A journalist writes news stories.",
   "He draws a map of the town.",
   "She makes a cup of tea.",
   "A bee collects nectar from flowers.",
   "He swims across the river.",
   "The sun sets in the west.",
   "She practices yoga in the morning.",
   "A plumber fixes a leaking tap.",
   "He watches the stars at night.",
   "Children build sandcastles at the beach.",
   "She arranges flowers in a vase.",
   "The baker kneads the dough.",
   "A pilot navigates the airplane.",
   "He climbs the mountain with enthusiasm.",
   "A cat lounges in the sun.",
   "She solves puzzles in her free time.",
   "A gardener waters the plants.",
   "The moon waxes and wanes.",
   "He rides his bike to school.",
   "A bird sings a melodious song.",
   "She whispers a secret.",
   "A fish jumps out of the water.",
   "He shovels snow from the driveway.",
   "A doctor prescribes medicine.",
   "She composes a new song.",
   "The sun illuminates the valley.",
   "A baby giggles when amused.",
   "He walks his dog every evening.",
   "A runner sprints to the finish line.",
   "She sketches a portrait.",
   "The leaves rustle in the wind.",
   "She enjoys walking in the forest.",
   "The bus arrives at the station.",
   "He watches television after dinner.",
   "The cat curls up on the sofa.",
   "Birds chirp early in the morning.",
   "She writes in her diary at night.",
   "The train speeds through the countryside.",
   "She sips her coffee slowly.",
   "He jogs around the track.",
   "The sun warms the sand.",
   "He feeds the ducks in the pond.",
   "She picks fresh berries from the bush.",
   "He reads the newspaper in the morning.",
   "The boat sails on the lake.",
   "She bakes a pie for dessert.",
   "He practices his speech.",
   "The stars shine brightly tonight.",
   "She collects seashells on the beach.",
   "He changes the oil in his car.",
   "The teacher assigns homework.",
   "She knits a scarf for winter.",
   "He replaces a broken window.",
   "The flowers bloom in spring.",
   "She packs her suitcase for a trip.",
   "He flies a kite on a windy day.",
   "The kids build a snowman.",
   "She listens to classical music.",
   "He climbs a ladder to fix the roof.",
   "The dog barks at the mailman.",
   "She folds the laundry.",
   "He mows the lawn on Saturday.",
   "The moon casts shadows on the ground.",
   "She plants a vegetable garden.",
   "He assembles a bookshelf.",
   "The car needs a wash.",
   "She sets the table for dinner.",
   "He tunes his guitar.",
   "The baby claps her hands.",
   "She sews a button on her shirt.",
   "He watches a documentary.",
   "The cat catches a mouse.",
   "She decorates the room for the party.",
   "He shuffles the deck of cards.",
   "The kids play hide and seek.",
   "She makes a salad for lunch.",
   "He paints a fence.",
   "The wind whistles through the trees.",
   "She learns a new language.",
   "He practices karate.",
   "The clock strikes midnight.",
   "She writes a letter to a friend.",
   "He cleans the pool.",
   "The birds migrate in the fall.",
   "She rides a horse on the trail.",
   "He assembles a model airplane.",
   "The sun sets over the mountains.",
   "She roasts marshmallows over the fire.",
   "He tunes the piano.",
   "The children chase butterflies.",
   "She makes a sandcastle at the beach.",
   "He plants a tree in the yard.",
   "The cat stretches in the sun.",
   "She draws a portrait.",
   "He fixes a leaky faucet.",
   "The kids play in the sprinkler.",
   "She designs a website.",
   "He plays chess with his friend.",
   "The dog digs a hole in the garden.",
   "She bakes cookies for the bake sale.",
   "He repairs a bicycle.",
   "The clock chimes on the hour.",
   "She practices ballet.",
   "He builds a campfire.",
   "The fish swim in the aquarium.",
   "She arranges a bouquet of flowers.",
   "He studies astronomy.",
   "The kids jump in puddles.",
   "She prepares a presentation for work.",
   "He repairs a broken chair.",
   "The sun peeks through the clouds.",
   "She takes a photograph of the sunset.",
   "He plays the violin in an orchestra.",
   "The dog wags its tail.",
   "She designs a new dress.",
   "He whittles a piece of wood.",
   "The birds build a nest in the spring.",
   "She makes a cup of hot chocolate.",
   "He sets up a tent for camping.",
   "The children swing on the playground.",
   "She organizes her bookshelf.",
   "He sharpens his kitchen knives.",
   "The cat naps in the afternoon.",
   "She paints a landscape.",
   "He checks the weather forecast.",
   "The children sled down the hill.",
   "She makes a fruit smoothie.",
   "He adjusts the thermostat.",
   "The dog fetches the ball.",
   "She knits a pair of mittens.",
   "He polishes his shoes.",
   "The cat chases its tail.",
   "She plans a garden layout.",
   "He grills burgers on the barbecue.",
   "The kids play tag in the yard.",
   "She reads a novel by the fireplace.",
   "He sets up a new computer.",
   "The birds peck at the bird feeder.",
   "She mixes ingredients for a cake.",
   "He sharpens a pencil.",
   "The dog howls at the moon.",
   "She draws a bath.",
   "He checks the mailbox.",
   "The children color with crayons.",
   "She prunes the rose bushes.",
   "He plays a board game with family.",
   "The cat curls up in a basket.",
   "She prepares a pot of soup.",
   "He inspects the car engine.",
   "The kids have a pillow fight.",
   "She practices the flute.",
   "The sun shines brightly in the clear blue sky.",
   "Birds chirp melodiously in the lush green forest.",
   "She painted a beautiful landscape on the canvas.",
   "The library was quiet, filled with rows of books.",
   "He drove the car along the winding mountain road.",
   "Children played soccer in the park on a sunny day.",
   "She baked a delicious chocolate cake for her birthday.",
   "The computer programmer fixed the bug in the code.",
   "The gentle breeze swayed the tall trees in the backyard.",
   "They hiked to the top of the hill to watch the sunrise.",
   "The museum displayed ancient artifacts from various cultures.",
   "Farmers harvested crops in the vast, golden fields.",
   "The chef prepared a gourmet meal with fresh ingredients.",
   "Musicians played classical tunes at the concert hall.",
   "The astronaut shared experiences of space travel with students.",
   "Architects designed a sustainable building with modern technology.",
   "Tourists explored the historic city and its ancient monuments.",
   "He solved complex puzzles with ease and precision.",
   "The biologist studied exotic plants in the rainforest.",
   "Artists showcased their work at the downtown gallery.",
   "The teacher explained the laws of physics through experiments.",
   "Athletes competed in challenging sports at the international event.",
   "The journalist wrote insightful articles on current global issues.",
   "Engineers developed innovative solutions to environmental problems.",
   "The train journey offered breathtaking views of the countryside.",
   "The cat chased the mouse through the house.",
   "He walked his dog in the park every morning.",
   "She found a stray cat and decided to adopt it.",
   "The dog barked loudly at the stranger at the door.",
   "He trained his cat to do tricks.",
   "She went to the pet store to buy dog food.",
   "The cat sat on the windowsill, watching the birds.",
   "He took the dog for a long hike in the mountains.",
   "She brushed the cat's fur until it was smooth and shiny.",
   "The dog wagged its tail excitedly when it saw its owner.",
   "He built a small house for his cat in the backyard.",
   "She volunteered at the animal shelter, taking care of both cats and dogs.",
   "The cat curled up in her lap and fell asleep.",
   "He bought a new leash for walking his dog.",
   "She taught her cat to use a new type of litter box.",
   "The dog chased its tail in circles.",
   "He found a lost cat in the alley and returned it to its owner.",
   "She trained her dog to participate in agility competitions.",
   "The cat climbed to the top of the tree and watched the world below.",
   "He and his dog entered a pet talent show.",
   "She adopted a cat from her local animal rescue center.",
   "The dog howled at the moon late into the night.",
   "He used a laser pointer to play with his cat.",
   "She baked homemade dog treats for her pet.",
   "She took her cat to the vet for a routine check-up.",
   "The dog played fetch in the backyard, tirelessly chasing the ball.",
   "He taught his cat to respond to its name.",
   "She walked her dog along the beach, enjoying the sunset.",
   "The cat prowled around the garden, hunting for insects.",
   "He read a book about training dogs and learned new techniques.",
   "She prepared a special meal for her cat's birthday.",
   "The dog snuggled up next to him as he watched television.",
   "She found her cat napping in the sunniest spot of the house.",
   "He bought a new collar for his dog with a matching leash.",
   "The cat meowed loudly every morning to be fed.",
   "She entered her dog in a local pet parade.",
   "He built a climbing tree for his cat to play on.",
   "The dog's loud barking scared away the intruder.",
   "She brushed her cat's long fur to prevent it from matting.",
   "He took his dog to a training class to learn new commands.",
   "The cat chased after a laser light, darting across the room.",
   "She knitted a small sweater to keep her dog warm in winter.",
   "He set up a cozy bed for his cat near the fireplace.",
   "The dog eagerly awaited his return at the front door each day.",
   "She found her cat playing with a ball of yarn.",
   "He took his dog on a camping trip to the mountains.",
   "The cat climbed high up in the bookshelf, knocking down a few books.",
   "She taught her dog to fetch the newspaper each morning.",
   "She found her dog digging a hole in the garden.",
   "The cat jumped gracefully off the high fence.",
   "He spent the weekend building a new doghouse in the yard.",
   "She taught her cat to sit on command, much to everyone's surprise.",
   "The dog looked out the window, eagerly waiting for his owner to return.",
   "He captured a funny video of his cat chasing its own shadow.",
   "She went jogging every morning with her energetic dog.",
   "The cat seemed fascinated by the fish swimming in the aquarium.",
   "He made a small playground for his dog with various toys.",
   "She created a comfortable sleeping area for her cat by the window.",
   "The dog whined excitedly when it heard the sound of the food bag.",
   "He carefully groomed his cat, removing all the tangles from its fur.",
   "She trained her dog to do a high-five as a party trick.",
   "The cat watched curiously as birds chirped outside the window.",
   "He set up a series of agility courses for his dog in the backyard.",
   "She laughed as her cat playfully pounced on a stuffed toy.",
   "The dog enjoyed swimming in the lake, fetching sticks from the water.",
   "He bought a new scratching post to keep his cat entertained.",
   "She arranged a playdate for her dog with a neighbor's pet.",
   "The cat climbed into the cozy hammock and dozed off.",
   "He took his dog on peaceful walks in the forest trail.",
   "She captured adorable photos of her cat snuggled in blankets.",
   "The dog patiently waited for his treat after performing a trick.",
   "He watched a documentary about cat behavior to better understand his pet.",
]

similar_words = [
                 ('cat', 'dog'),
               #   ('car', 'vehicle'), 
               #   ('happy', 'joyful'),
                 ('she', 'he'),
               #   ('rug', 'mat'),
               #   ('girl', 'boy'),
               #   ('cake', 'pie'),
               #   ('boat','vehicle'),
               #   ('train', 'vehicle'),
               #   ('sun', 'moon'),
               #   ('morning', 'night'),
                 ]



# Tokenizer training
def train_tokenizer(sentences, token_fn):
   tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
   trainer = trainers.BpeTrainer(special_tokens=["[UNK]"])
   tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
   tokenizer.train_from_iterator(sentences, trainer)
   tokenizer.save(token_fn)

# Load or train tokenizer
if os.path.exists(token_fn):
   tokenizer = Tokenizer.from_file(token_fn)
else:
   train_tokenizer(sentences, token_fn)
   tokenizer = Tokenizer.from_file(token_fn)

# Custom Dataset
class TextDataset(Dataset):
   def __init__(self, sentences, tokenizer, context_size):
      self.data = []
      for sentence in sentences:
         tokenized_output = tokenizer.encode(sentence)
         input_ids = tokenized_output.ids
         for i in range(len(input_ids) - context_size):
            context = input_ids[i:i + context_size]
            target = input_ids[i + context_size]
            self.data.append((context, target))

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
   checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:06}_step_{step:06}.pth')
   torch.save({
      'epoch': epoch,
      'step': step,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
   }, checkpoint_path)

# Function to load the latest model checkpoint
def load_latest_checkpoint(checkpoint_dir):
   checkpoints = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pth')]
   latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
   print(f"Loading checkpoint: {latest_checkpoint}")
   return torch.load(os.path.join(checkpoint_dir, latest_checkpoint))#, map_location=torch.device('cpu'))


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

def plot_embeddings2(model, tokenizer, epoch, sentences, plot_dir='plots'):
   # Ensure the plotting directory exists
   os.makedirs(plot_dir, exist_ok=True)

   # Tokenize sentences and get unique words
   unique_words = set()
   for sentence in sentences:
      tokens = tokenizer.encode(sentence).tokens
      unique_words.update(tokens)

   # Extract embeddings
   embeddings = model.embeddings.weight.detach().cpu().numpy()
   vocab = tokenizer.get_vocab()

   # Plotting the embeddings for words in the sentences
   plt.figure(figsize=(10, 8))
   for word in unique_words:
      if word in vocab:
         word_index = vocab[word]
         plt.scatter(embeddings[word_index, 0], embeddings[word_index, 1])
         plt.text(embeddings[word_index, 0], embeddings[word_index, 1], word, fontsize=9)
   plt.xlabel('Dimension 1')
   plt.ylabel('Dimension 2')
   plt.title(f'Word Embeddings at Epoch {epoch+1}')
   plt.savefig(os.path.join(plot_dir, f'embeddings_epoch_{epoch+1}.png'))
   plt.savefig(os.path.join(plot_dir, f'embeddings_epoch_{epoch+1}.pdf'))
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
    print(vec1.shape,vec2.shape)
    return cosine_similarity(vec1, vec2).item()

# Training loop
def train(model, data_loader, optimizer, criterion, scheduler, epochs, print_every, save_interval, checkpoint_dir, log_dir, starting_epoch=0):
   writer = SummaryWriter(log_dir)
   model.train()
   for epoch in range(starting_epoch,starting_epoch+epochs):
      total_loss = 0.0
      for step, (context, target) in enumerate(data_loader):
         context, target = context.to(device), target.to(device)
         optimizer.zero_grad()
         log_probs = model(context)
         loss = criterion(log_probs, target)
         loss.backward()
         optimizer.step()
         scheduler.step()
         total_loss += loss.item()

         if (step + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs+starting_epoch}], Step [{step+1}/{len(data_loader)}], Loss: {loss.item():.4e}")
            writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader) + step)
            # Log the current learning rate
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Learning Rate', current_lr, epoch * len(data_loader) + step)

      if (epoch + 1) % save_interval == 0:
         save_model(model, optimizer, epoch, step, checkpoint_dir)
      
      if (epoch + 1) % PLOT_EVERY == 0:
         plot_embeddings_with_cosine_similarity(model, tokenizer, epoch, similar_words)

         embeddings = model.embeddings.weight.data
         for word1, word2 in similar_words:
            if word1 in tokenizer.get_vocab() and word2 in tokenizer.get_vocab():
               index1 = tokenizer.token_to_id(word1)
               index2 = tokenizer.token_to_id(word2)
               similarity = compute_cosine_similarity(embeddings, index1, index2)
               writer.add_scalar(f'Cosine Similarity/{word1}-{word2}', similarity, epoch)

      avg_loss = total_loss / len(data_loader)
      writer.add_scalar('Average Loss', avg_loss, epoch)
      # Log the current learning rate
      current_lr = scheduler.get_last_lr()[0]
      writer.add_scalar('Learning Rate', current_lr, (epoch+1) * len(data_loader))
      print(f"Epoch [{epoch+1}/{epochs+starting_epoch}] completed, Total Loss: {avg_loss:.4e}")

   writer.close()

# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.get_vocab_size()
model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE).to(device)
checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_*.pth')))
starting_epoch = 0
if len(checkpoints) > 0:
   checkpoint = load_latest_checkpoint(CHECKPOINT_DIR)
   starting_epoch = checkpoint['epoch'] + 1
   model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, step_size_up=20000, mode='triangular2')
criterion = nn.NLLLoss()
dataset = TextDataset(sentences, tokenizer, CONTEXT_SIZE)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

train(model, data_loader, optimizer, criterion, scheduler, EPOCHS, PRINT_EVERY, SAVE_INTERVAL, CHECKPOINT_DIR, LOG_DIR, starting_epoch)
