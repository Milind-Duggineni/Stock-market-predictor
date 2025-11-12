# utils/timer.py
import time
from functools import wraps

def timer(func):
    """Decorator to measure runtime of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[TIMER] {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Example usage
if __name__ == "__main__":
    @timer
    def example():
        total = sum(range(1000000))
        return total
    example()
