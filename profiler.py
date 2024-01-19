import torch


def get_profiler(log_dir,
                 activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]):
   # Defining the profiler
   prof = torch.profiler.profile(
         activities=activities,
         #schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
         on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
         #record_shapes=True,
         #profile_memory=True,
         #with_stack=True,
         #with_flops=True,
         #with_modules=True
         )
   return prof

