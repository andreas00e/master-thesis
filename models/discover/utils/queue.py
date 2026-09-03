from termcolor import colored

import torch 
import torch.nn as nn 
from torchtyping import TensorType


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
        self._is_full = False 
    
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
        if self.write_idx + num_elements < self.capacity and  self.queue_elements < self.capacity: 
            self.queue[:, self.write_idx:self.write_idx+num_elements].copy_(x)
            self.write_idx = (self.write_idx + num_elements) % self.capacity
            self.queue_elements += num_elements 
        # Overflow 
        else: 
            overflow = (self.queue_elements + num_elements) - self.capacity # how many elements must go 
            queue_buf = self.queue.clone()[overflow:] 
            self.queue[:, -num_elements:] = x 
            self.queue[:, :overflow] = queue_buf
            
            self.write_idx = self.capacity
            self.queue_elements = self.capacity  
            self._is_full = True
    
    @torch.no_grad()
    def dequeue(self) -> torch.Tensor: 
        return self.queue
    
    @property
    def is_full(self) -> bool: 
        return self._is_full