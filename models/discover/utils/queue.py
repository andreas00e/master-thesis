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
        
        self.register_buffer("queue", torch.zeros(size=(self.num_modalities, self.capacity, feature_dim), dtype=dtype, device=device))
        self.write_idx = 0 
        self.queue_elements = 0 
    
    @torch.no_grad()
    def enqueue(self, x: TensorType["num_modalities", "num_elements", "feature_dim"]) -> None:        
        if x.ndim != 3: 
            raise ValueError(colored(f"Expected x to have shape [num_modalities, num_elements, feature_dim, but got {tuple(x.shape)}."), "red")
        
        num_modalities, num_elements, feature_dim = x.shape
        
        if num_modalities != self.num_modalities: 
            raise ValueError(colored(f"Expected num_modalities {self.num_modalities}, got {num_modalities}."), "red")        
        if feature_dim != self.feature_dim: 
            raise ValueError(colored(f"Expected feature_dim {self.feature_dim}, got {feature_dim}."), "red")
        if num_elements > self.capacity: 
            raise ValueError(colored(f"The queue's maximal capacity is {self.capacity}, but got input size {x.shape[0]}.", "red"))
        if num_elements == 0: 
            return
        
        # Input fits into queue entirely 
        if self.write_idx + num_elements <= self.capacity : 
            self.queue[:, self.write_idx:self.write_idx+num_elements].copy_(x)
        # Overflow 
        else: 
            first_part = self.capacity - self.write_idx
            second_part = num_elements - first_part

            self.queue[:, self.write_idx:self.capacity].copy_(x[:, :first_part]) 
            self.queue[:, :second_part].copy_(x[:, first_part:])
            
            self.write_idx = (self.write_idx + num_elements) % self.capacity
            self.queue_elements = min(self.capacity, self.queue_elements + num_elements)
    
    @torch.no_grad()
    def dequeue(self) -> torch.Tensor: 
        return self.queue[:, :self.queue_elements]
    
    @property
    def is_full(self) -> bool: 
        return self.queue_elements == self.capacity