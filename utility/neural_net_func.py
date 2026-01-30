import numpy as np
import torch

from core.config import Settings


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
