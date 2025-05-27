import time
from typing import Callable, List
import torch

def benchmark(description: str, run: Callable[[], None], num_warmups: int = 1, num_trials: int = 3) -> List[float]:
    """
    Benchmark a function by running it multiple times and return execution times in milliseconds.
    """
    for _ in range(num_warmups):
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  
    times: List[float] = []
    for _ in range(num_trials):
        start_time = time.time()
        
        run()  
        if torch.cuda.is_available():
            torch.cuda.synchronize()  
            
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000  
        times.append(elapsed_ms)
    
    return times


if __name__ == "__main__":
    def test_function():
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = x @ y  
        return z
    
    execution_times = benchmark(
        description="GPU matrix multiplication",
        run=test_function,
        num_warmups=2,
        num_trials=5
    )
    
    print(f"Execution times (ms): {execution_times}")
    print(f"Average time: {sum(execution_times)/len(execution_times):.2f} ms")
    print(f"Best time: {min(execution_times):.2f} ms")
    print(f"Worst time: {max(execution_times):.2f} ms")