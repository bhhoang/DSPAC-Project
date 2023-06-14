import os

# Os absolute path ./dataset
dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))

# export dataset
__all__ = [dataset]

