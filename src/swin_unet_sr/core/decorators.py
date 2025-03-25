import functools
import time
from typing import Callable


def time_this(callable: Callable) -> Callable:
    @functools.wraps(callable)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = callable(*args, **kwargs)
        t_elapsed = 1000 * (time.perf_counter() - t0)

        print(f"Time this: <{callable.__qualname__}> takes {t_elapsed:.4f} ms")

        return result

    return wrapper


def show_info(callable: Callable) -> Callable:
    @functools.wraps(callable)
    def wrapper(*args, **kwargs):
        print("=" * 50)
        print(f"Calling [{callable.__qualname__}]...")
        result = callable(*args, **kwargs)
        print("Finished")
        print("=" * 50)
        print()

        return result

    return wrapper
