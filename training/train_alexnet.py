import torch
from torch import optim, nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import click
import os

from utility.load_data import CustomData
from utility.transform_data import transform_data
from utility.neural_net_func import train
from core.config import Settings

class AlexNet(nn.Module):
    def __init__(self, 
                 idx2class: dict, 
                 flatten_in_features: int):
        super(AlexNet, self).__init__()
        self.idx2class = idx2class
        self.in_feat = flatten_in_features
        self.feats = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.clf = nn.Linear(in_features=self.in_feat, out_features=len(self.idx2class))
    def forward(self, inp):
        op = self.feats(inp)
        op = op.view(op.size(0), -1)
        op = self.clf(op)
        return op
 
def flatten_feature_calculation(
        image_size:int=224) -> int:
    def CNN_out(input_, kernel, padding, stride):
        return np.floor((input_-kernel+2*padding)/stride)+1
    cn1_out = CNN_out(image_size, 11, 5, 4)/2
    cn2_out = CNN_out(cn1_out, 5, 2, 1)/2
    cn3_out = CNN_out(cn2_out, 3, 1, 1)
    cn4_out = CNN_out(cn3_out, 3, 1, 1)
    cn5_out = CNN_out(cn4_out, 3, 1, 1)
    return cn5_out*cn5_out*256

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
                         transform=transform_data(augemntation=augmentation_bool)
                         )
    train_dataset, val_dataset = random_split(dataset, [int(test_split*len(dataset)), len(dataset)-int(test_split*len(dataset))])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    alexnet = AlexNet(dataset.get_classes()[1], 
                      flatten_in_features=flatten_feature_calculation(
                          image_size=image_size
                        )
                    )
    alexnet.to(Settings.device)

    optimiser = optim.AdamW(alexnet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loss, train_accuracy, val_loss, val_accuracy = train(
        model=alexnet, trainloader=train_loader, valloader=val_loader, epochs=epochs, optimiser=optimiser, criterion=criterion
    )
    print("\n")
    print("===================================")
    print("\n")
    print(f"Training Accuracy: {train_accuracy}")
    print("\n")
    print("===================================")
    print(f"Validation Accuracy: {val_accuracy}")

    print("\n")
    input("Press any key to save the model:")
    print("\n")
    filename = str(input("Enter the filename for the model:"))
    filename = filename if filename[-4:] == ".pth" else filename+".pth"
    full_path = os.path.join(Settings.root_dir, Settings.saved_model_path)
    full_path = os.path.join(full_path, filename)
    torch.save(alexnet.state_dict(), full_path)
    print(f"Model weights saved to {full_path}")

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