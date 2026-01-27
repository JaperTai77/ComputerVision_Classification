import torch
from torch import optim, nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import click
import os

from utility.load_data import CustomData
from utility.transform_data import transform_data
from core.config import Settings

class ModifiedLeNet(nn.Module):
    def __init__(self, 
                 idx2class: dict, 
                 flatten_in_features: int, 
                 kernel: int=5, 
                 cnn1_out_channel: int=6, 
                 cnn2_out_channel: int=16):
        super().__init__()
        self.idx2class = idx2class
        self.in_feat = flatten_in_features
        self.kernel = kernel
        self.cnn1_out_channel = cnn1_out_channel
        self.cnn2_out_channel = cnn2_out_channel
        self.cn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.cnn1_out_channel, kernel_size=kernel, stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.cn2 = nn.Sequential(
            nn.Conv2d(in_channels=self.cnn1_out_channel, out_channels=self.cnn2_out_channel, kernel_size=kernel, stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_feat, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, len(self.idx2class)),
        )

    def forward(self, x):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
def flatten_feature_calculation(
        image_size:int=224, 
        kernel: int=5, 
        cnn2_out_channel: int=16):
    def CNN_out(input_, kernel, padding, stride):
        return np.floor((input_-kernel+2*padding)/stride)+1
    cn1_out = CNN_out(image_size, kernel, 0, 1)/2
    cn2_out = CNN_out(cn1_out, kernel, 0, 1)/2
    return cn2_out*cn2_out*cnn2_out_channel

def train(model, trainloader, valloader, epochs:int, optimiser, criterion)->tuple[list, list, list, list]:
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []
        
        loss_total = 0.0; total = 0; correct = 0
        for _, data in enumerate(trainloader):
            img, label = data
            img = img.to(Settings.device)
            label = label.to(Settings.device)
            
            optimiser.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimiser.step()

            loss_total += loss.item()*label.size(0)
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
        
        train_accuracy.append(correct/total)
        train_loss.append(loss_total/len(trainloader))
        print(f'Epoch {epoch+1} Training Accuracy = {correct/total}')
        print(f'Epoch {epoch+1} Training Loss = {loss_total/len(trainloader)}')
    
        # validation
        if epoch%1 == 0:
            model.eval()
            loss_total = 0.0; total = 0; correct = 0

            with torch.no_grad():
                for img, label in valloader:
                    img = img.to(Settings.device)
                    label = label.to(Settings.device)
                    output = model(img)
                    loss = criterion(output, label)
                    
                    loss_total += loss.item()*label.size(0)
                    _, pred = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (pred == label).sum().item()

                val_accuracy.append(correct/total)
                val_loss.append(loss_total/len(valloader))

                print(f'Epoch {epoch+1} Validation Accuracy = {correct/total}')
                print(f'Epoch {epoch+1} Validation Loss = {loss_total/len(valloader)}')
    
    return train_loss, train_accuracy, val_loss, val_accuracy


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

    lenet = ModifiedLeNet(dataset.get_classes()[1], flatten_in_features=flatten_feature_calculation(image_size=image_size))
    lenet.to(Settings.device)

    optimiser = optim.AdamW(lenet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loss, train_accuracy, val_loss, val_accuracy = train(
        model=lenet, trainloader=train_loader, valloader=val_loader, epochs=epochs, optimiser=optimiser, criterion=criterion
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
    torch.save(lenet.state_dict(), full_path)
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

