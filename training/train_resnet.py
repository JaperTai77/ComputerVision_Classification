import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
import click
import os
import torchvision.models as models

from utility.load_data import CustomData
from utility.transform_data import transform_data
from utility.neural_net_func import train
from core.config import Settings

class ResNet18(nn.Module):
    def __init__(self, idx2class: dict):
        super(ResNet18, self).__init__()
        self.idx2class = idx2class
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the final layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, len(self.idx2class))

    def forward(self, x):
        return self.resnet(x)

def main(root_dir:str,
         train_csv:str,
         target:str,
         augmentation_bool:bool,
         image_size:int,
         test_split:float,
         batch_size:int,
         epochs:int):
    dataset = CustomData(root_dir=root_dir,
                         train_csv=train_csv,
                         target_col_name=target,
                         transform=transform_data(augmentation=augmentation_bool, image_size=image_size)
                         )

    train_size = int(test_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    resnet = ResNet18(dataset.get_classes()[1])
    resnet.to(Settings.device)

    optimiser = optim.AdamW(resnet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loss, train_accuracy, val_loss, val_accuracy = train(
        model=resnet, trainloader=train_loader, valloader=val_loader, epochs=epochs, optimiser=optimiser, criterion=criterion
    )

    print("\n")
    print("===================================")
    print("\n")
    print(f"Training Accuracy: {train_accuracy}")
    print("\n")
    print("===================================")
    print(f"Validation Accuracy: {val_accuracy}")

    print("\n")

    # Save model logic
    try:
        if os.isatty(0): # Check if interactive
            input("Press any key to save the model:")
            print("\n")
            filename = str(input("Enter the filename for the model:"))
        else:
            # Default behavior for non-interactive
            filename = "resnet18_model.pth"
            print(f"Non-interactive mode detected. Saving to {filename}")

        filename = filename if filename[-4:] == ".pth" else filename+".pth"

        # Ensure directory exists
        save_dir = os.path.join(Settings.root_dir, Settings.saved_model_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        full_path = os.path.join(save_dir, filename)
        torch.save(resnet.state_dict(), full_path)
        print(f"Model weights saved to {full_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

@click.command()
@click.option("-a", "--augmentation", "a", help="Whether to apply augmentation to our training data", default=False, type=bool)
@click.option("-i", "--imagesize", "i", help="The image size that use to train the data", default=224, type=int)
@click.option("-s", "--trainsplitratio", "s", help="The ratio of the data that is used for training", default=0.8, type=float)
@click.option("-b", "--batchsize", "b", help="The batch size", default=16, type=int)
@click.option("-e", "--epochs", "e", help="Training epochs", default=5, type=int)
def run(a, i, s, b, e):
    print("Loading data...")
    print("\n")
    main(root_dir=Settings.root_dir,
         train_csv=Settings.train_csv,
         target=Settings.target_col_name,
         augmentation_bool=a,
         image_size=i,
         test_split=s, batch_size=b, epochs=e
    )

if __name__ == "__main__":
  #import warnings
  #warnings.filterwarnings("ignore")
  run()
