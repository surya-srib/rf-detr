from detr import ViTLarge
import torch
import argparse

path = "<path>"
epochs_count = 12

parser = argparse.ArgumentParser("ViT train parser")
parser.add_argument('--path', type=str)
parser.add_argument('--epochs', type=int)

args = parser.parse_args()

if args.path:
    path = args.path

if args.epochs:
    epochs_count = args.epochs

print(path)
print(epochs_count)

if __name__ == "__main__":
    model = ViTLarge()
    model.train(
        dataset_dir=path,
        epochs=epochs_count,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
