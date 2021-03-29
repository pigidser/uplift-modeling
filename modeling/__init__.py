from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    """
    Allows to suppress unwilling output

    print("Now you see it")
    with suppress_stdout():
        print("Now you don't")
    
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
