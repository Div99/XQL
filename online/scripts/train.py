import argparse
from research.utils.trainer import Config, train

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    args = parser.parse_args()

    config = Config.load(args.config)
    train(config, args.path, device=args.device)