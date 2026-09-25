from termcolor import colored

import torch 
import torch.nn as nn 
from torchtyping import TensorType

# Circular FIFO Queue
class FIFOQueue(nn.Module): 
    def __init__(
        self, 
        num_modalities: int, 
        capacity: int, 
        feature_dim: int, 
        dtype: torch.dtype = torch.float32, 
        device: str="cuda"
        ) -> None:
        super().__init__()
        
        if num_modalities <= 0: 
            raise ValueError(colored(f"num_modalities must be positive, but got {num_modalities}.", "red"))
        if capacity <= 0: 
            raise ValueError(colored(f"capacity must be positive, but got {capacity}.", "red"))
        if feature_dim <= 0: 
            raise ValueError(colored(f"feature_dim must be positive, but got {feature_dim}.", "red"))
        
        self.num_modalities = num_modalities
        self.capacity = capacity
        self.feature_dim = feature_dim
        
        self.register_buffer("queue", torch.zeros(size=(self.num_modalities, self.capacity, self.feature_dim), dtype=dtype, device=device))
        self.register_buffer("write_idx", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer("queue_elements", torch.tensor(0,  dtype=torch.long, device=device))
    
    @torch.no_grad()
    def enqueue(self, x: TensorType["num_modalities", "num_elements", "feature_dim"]) -> None:        
        if x.ndim != 3: 
            raise ValueError(colored(f"Expected x to have shape [num_modalities, num_elements, feature_dim], but got {tuple(x.shape)}."), "red")
        
        num_modalities, num_elements, feature_dim = x.shape
        
        if num_modalities != self.num_modalities: 
            raise ValueError(colored(f"Expected num_modalities {self.num_modalities}, got {num_modalities}."), "red")        
        if feature_dim != self.feature_dim: 
            raise ValueError(colored(f"Expected feature_dim {self.feature_dim}, got {feature_dim}."), "red")
        if num_elements > self.capacity: 
            raise ValueError(colored(f"The queue's maximal capacity is {self.capacity}, but got input size {x.shape[1]}.", "red"))
        if num_elements == 0: 
            return
        
        write_idx = self.write_idx.item()
        queue_elements = self.queue_elements.item()

        # Queue is not full yet => input fits into queue entirely 
        # or
        # Queue is already full => old elements are overwritten/ "eaten" first
        if write_idx + num_elements <= self.capacity: 
            self.queue[:, write_idx:write_idx+num_elements].copy_(x)
        # Overflow 
        else: 
            first_part = self.capacity - write_idx # number of elements added at the end of the queue
            second_part = num_elements - first_part # number of elements added at the beginning of the queue

            self.queue[:, write_idx:].copy_(x[:, :first_part]) 
            self.queue[:, :second_part].copy_(x[:, first_part:])
            
        self.write_idx.copy_((write_idx + num_elements) % self.capacity)
        self.queue_elements.copy_(min(self.capacity, queue_elements + num_elements))
    
    @torch.no_grad()
    def get(self) -> torch.Tensor: 
        write_idx = int(self.write_idx.item())
        queue_elements = int(self.queue_elements.item())
        
        if self.is_full: 
            first_part = self.queue[:, write_idx:]
            second_part = self.queue[:, :write_idx]
            return torch.cat((first_part, second_part), dim=1)
        else:
            return self.queue[:, :queue_elements].clone()
    
    @property
    def is_full(self) -> bool: 
        return int(self.queue_elements.item()) == self.capacity
    
    @torch.no_grad()
    def reset(self) -> None:
        self.queue.zero_()
        self.write_idx.zero_()
        self.queue_elements.zero_()