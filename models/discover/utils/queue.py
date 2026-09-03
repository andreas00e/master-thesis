from termcolor import colored

import torch 
import torch.nn as nn 
from torchtyping import TensorType


class FIFOQueue(nn.Module): 
    def __init__(
        self, 
        capacity: int, 
        feature_dim: int, 
        dtype: torch.dtype = torch.float32, 
        device: str="cuda"
        ) -> None:
        super().__init__()
        
        if capacity <= 0: 
            raise ValueError(colored(f"max_elements must be positive, got {capacity}.", "red"))
        if feature_dim <= 0: 
            raise ValueError(colored(f"feature_dim must be positive, got {feature_dim}.", "red"))
        
        self.capacity = capacity
        self.feature_dim = feature_dim
        
        self.register_buffer("queue", torch.zeros(size=(self.capacity, feature_dim), dtype=dtype, device=device))
        self.write_idx = 0 
        self.queue_elements = 0 
    
    @torch.no_grad()
    def enqueue(self, x: TensorType["num_elements", "feature_dim"]) -> None:
        num_elements = x.shape[0]
        
        if x.ndim != 2: 
            raise ValueError(colored(f"Expected x to have shape [N, {self.feature_dim}], got {tuple(x.shape)}."), "red")
        if x.shape[1] != self.feature_dim: 
            raise ValueError(colored(f"Expected feature_dim {self.feature_dim}, got {x.shape[1]}."), "red")
        if num_elements > self.capacity: 
            raise ValueError(colored(f"The queue's maximal capacity is {self.capacity}, but got input size {x.shape[0]}.", "red"))
        if num_elements == 0: 
            return
        
        # Input fits into queue entirely 
        if self.write_idx + num_elements < self.capacity and  self.queue_elements < self.capacity: 
            self.queue[self.write_idx:self.write_idx+num_elements].copy_(x)
            self.write_idx = (self.write_idx + num_elements) % self.capacity
            self.queue_elements += num_elements 
        # Overflow 
        else: 
            overflow = (self.queue_elements + num_elements) - self.capacity # how many elements must go 
            queue_buf = self.queue.clone()[overflow:] 
            self.queue[-num_elements:] = x 
            self.queue[:overflow] = queue_buf
            
            self.write_idx = self.capacity
            self.queue_elements = self.capacity  
            
    def get_all(self) -> TensorType["n", "feature_dim"]:
        return self.queue