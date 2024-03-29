import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from model import save_checkpoint,load_checkpoint
import numpy as np
import profiler
import os
import socket
import datetime,time
import logging
logger = logging.getLogger(__name__)
from mpi import COMM_WORLD,rank,world_size,local_rank

def setup(rank, world_size):
   if rank == 0:
      master_addr = socket.gethostname()
      sock = socket.socket()
      sock.bind(('',0))
      master_port  = sock.getsockname()[1]
      master_port  = 2345
   else:
      master_addr = None
      master_port = None
   master_addr = COMM_WORLD.bcast(master_addr, root=0)
   master_port = COMM_WORLD.bcast(master_port, root=0)
   # print(f'rank {rank} master_addr: {master_addr}, master_port: {master_port}')
   os.environ["MASTER_ADDR"] = master_addr
   os.environ["MASTER_PORT"] = str(master_port)
   # dist.init_process_group("mpi", rank=rank, world_size=world_size)
   backend = 'gloo'
   init_method = 'env://'
   torch.distributed.init_process_group(
         backend     = backend,
         init_method = init_method,
         world_size  = world_size,
         rank        = rank,
         timeout     = datetime.timedelta(seconds=120)
   )
   COMM_WORLD.barrier()

def cleanup():
   torch.distributed.destroy_process_group()

def lr_schedule(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        # Cosine decay
        return 0.5 * (1.0 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

def lr_scheduleA(step, warmup_steps, total_steps, decay_factor=0.5):
   if step < warmup_steps:
      return float(step) / float(max(1, warmup_steps))
   else:
      # decay factor that decreases each cosine period (every 2 pi)
      factors_of_pi = (step - warmup_steps) / (total_steps - warmup_steps)
      decay_factor = decay_factor ** max(int((int(factors_of_pi)-1)/2),0)
      return decay_factor * 0.5 * (1.0 + np.cos(np.pi * factors_of_pi))



def train(model, dataset, epochs, lr, batch_size, log_interval, device, output_dir,
          pytorch_profiler=False,profile_steps=1000,warmup_steps=5000,total_steps=1e4):

   CHECKPOINT_DIR = os.path.join(output_dir, "checkpoints")
   LOG_DIR = os.path.join(output_dir, "tensorboard")
   if rank == 0:
      os.makedirs(output_dir, exist_ok=True)
      os.makedirs(CHECKPOINT_DIR, exist_ok=True)
      os.makedirs(LOG_DIR, exist_ok=True)
   
   logging.debug('Starting training')
   setup(rank, world_size)

   similar_words = [("he", "she"),
                    ("king", "queen"),
                    ("man", "woman"),
                    ("a","an"),
                    ("them","they"),
                    ("")]

   # Only write logs from the first process to avoid duplicate logs
   writer = SummaryWriter(log_dir=LOG_DIR) if rank == 0 else None

   # Create a distributed sampler and loader
   logging.debug('Creating distributed sampler and loader')

   # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
   # data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)#,num_workers=4,pin_memory=True,persistent_workers=True)
   data_loader = DataLoader(dataset, batch_size=batch_size)

   # Wrap model for distributed training
   logging.debug('Wrapping model for distributed training')
   model = DDP(model, device_ids=[local_rank])

   loss_fn = nn.NLLLoss()
   optimizer = optim.Adam(model.parameters(), lr = lr)

   # Define the LR scheduler
   scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_scheduleA(step, warmup_steps, total_steps))


   prof = None
   if pytorch_profiler:
      logger.info('running pytorch profiler')
      prof = profiler.get_profiler(LOG_DIR)

   for epoch in range(epochs):
      model.train()
      total_loss = torch.zeros(1, device=device)
      logging.info(f'Starting Epoch {epoch}:')
      rate_timer_start = time.time()
      
      if prof:
         prof.start()
      for i, (context, target) in enumerate(data_loader):
         # Move data to the current device
         if prof: torch.cuda.synchronize()
         context, target = context.to(device), target.to(device)
         if prof: torch.cuda.synchronize()
         model.zero_grad()
         if prof: torch.cuda.synchronize()
         output = model(context)
         if prof: torch.cuda.synchronize()
         loss = loss_fn(output, target)
         if prof: torch.cuda.synchronize()
         loss.backward()
         if prof: torch.cuda.synchronize()
         optimizer.step()
         if prof: torch.cuda.synchronize()

         # Update the learning rate
         scheduler.step()
         
         total_loss += loss.detach()
         if writer and ((i+1) % log_interval == 0) and rank == 0:
            avg_rate =  log_interval * batch_size * world_size / (time.time() - rate_timer_start)
            avg_loss = total_loss.item() / log_interval
            rate_timer_start = time.time()
            writer.add_scalar('Metrics/Running Average Loss', avg_loss, epoch * len(data_loader) + i)
            writer.add_scalar('Metrics/Running Average Rate', avg_rate, epoch * len(data_loader) + i)
            # writer.add_scalar('Metrics/Learning Rate', optimizer.param_groups[0]['lr'], epoch * len(data_loader) + i)
            writer.add_scalar('Metrics/Learning Rate', scheduler.get_last_lr()[0], epoch * len(data_loader) + i)
            writer.add_scalar('Metrics/Epoch', epoch, epoch * len(data_loader) + i)
            # print log output including loss and rate using scientific notation
            logging.info(f'Rank {rank}/{world_size} Epoch {epoch}/{epochs} Batch {i}/{len(data_loader)}: Loss {avg_loss:.4e} Rate {avg_rate:.4e} LR {scheduler.get_last_lr()}')
            total_loss = torch.zeros(1, device=device)

            if not dataset.use_synthetic:
               with torch.no_grad():
                  word_embeddings = model.module.embeddings.weight
                  for word1, word2 in similar_words:
                     idx1 = dataset.tokenizer.encode(word1).ids[0]
                     idx2 = dataset.tokenizer.encode(word2).ids[0]

                     embed1 = word_embeddings[idx1]
                     embed2 = word_embeddings[idx2]

                     cos_sim = F.cosine_similarity(embed1.unsqueeze(0), embed2.unsqueeze(0))
                     writer.add_scalar(f'Cosine_Similarity/{word1}_{word2}', cos_sim, epoch * len(data_loader) + i)
         if prof:
            prof.step()
            if i >= profile_steps:
               break
      if prof:
         prof.stop()
         if rank == 0:
            print(prof.key_averages())
         break
      # save model once per epoch
      if rank == 0:
         save_checkpoint(model, optimizer, epoch+1,0, CHECKPOINT_DIR)

   if writer:
      writer.close()

   cleanup()

# # Example usage
# world_size = 4  # Number of GPUs
# epochs = 10
# lr = 0.001
# batch_size = 64
# log_interval = 10

# # Assume model and dataset are predefined
# for rank in range(world_size):
#    torch.multiprocessing.spawn(train, args=(world_size, model, dataset, epochs, lr, batch_size, log_interval), nprocs=world_size, join=True)
