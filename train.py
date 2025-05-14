import sys
from rl_zoo3.train import train

if __name__ == "__main__":
    sys.argv = ["train.py"] + sys.argv[1:]  # Add script name to match expected format
    train()
