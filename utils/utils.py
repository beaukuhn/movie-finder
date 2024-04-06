import time
import random


def throttle(func, delay=0.5):
    """
    Add a delay to a function.
    """

    def throttled_func(*args, **kwargs):
        time.sleep(delay)
        return func(*args, **kwargs)

    return throttled_func


def retry_with_exponential_backoff(max_retries=5, backoff_factor=0.5):
    """
    Decorator factory to retry a function with exponential backoff.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise

                    wait_time = backoff_factor * (2 ** (retries - 1))

                    # Adding jitter
                    wait_time += random.uniform(0, 0.1 * wait_time)
                    time.sleep(wait_time)

        return wrapper

    return decorator
