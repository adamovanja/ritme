def main_function(func):
    """Decorator to flag a function as a main function."""
    func.is_main = True
    return func


def helper_function(func):
    """Decorator to flag a function as a helper function."""
    func.is_helper = True
    return func
