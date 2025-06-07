import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training = True):
    """

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training:
        train_set = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', transform=transform, download=True),
            batch_size=64, shuffle=False)
        return train_set
    else:
        test_set = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transform, download = True),
            batch_size=64)
        return test_set


def build_model():
    """

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features= 784, out_features= 128, bias=True),
        nn.ReLU(),
        nn.Linear(in_features= 128, out_features= 64, bias=True),
        nn.ReLU(),
        nn.Linear(in_features= 64, out_features= 10, bias=True),
    )
    return model


def build_deeper_model():
    """

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features= 28*28, out_features= 256, bias=True),
        nn.ReLU(),
        nn.Linear(in_features= 256, out_features= 128, bias=True),
        nn.ReLU(),
        nn.Linear(in_features= 128, out_features= 64, bias=True),
        nn.ReLU(),
        nn.Linear(in_features= 64, out_features= 32, bias=True),
        nn.ReLU(),
        nn.Linear(in_features= 32, out_features= 10, bias=True)
    )
    return model

def train_model(model, train_loader, criterion, T):
    """

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(T):
        total, correct, totalLoss = 0, 0, 0
        for i, (images, labels) in enumerate(train_loader):
            # Zero parameter gradients
            opt.zero_grad()

            # Forward pass + Backward pass + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            totalLoss += loss.item()
            total += labels.size(0) 
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
           
        print(f"Train Epoch: {epoch}, Accuracy: {correct}/{total}({accuracy * 100:.2f}%), Loss: {loss.item():.3f}")
    return None

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    with torch.no_grad():
        runningLoss,total,correct = 0,0,0
        for i, (images, labels) in enumerate(test_loader):
            output = model(images)
            loss = criterion(output, labels)
            _, predicted = torch.max(output, 1)    
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            runningLoss += loss.item() * images.size(0)
        accuracy = correct / total
        average_loss = runningLoss / total

        if show_loss:
            print(f"Average loss: {average_loss:.4f}")
            print(f"Accuracy: {accuracy * 100:.2f}%")

        else:
            print(f"Accuracy: {accuracy * 100:.2f}%")
        
    return None

def predict_label(model, test_images, index):
    """

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    with torch.no_grad(): 
        image = test_images[index].unsqueeze(0) 
        outputs = model(image) 
        prob = F.softmax(outputs, dim= 1)

        top3_probs, top3_indices = torch.topk(prob, 3)  # Get top 3 predictions

        # top 3 predictrions
        for i in range(3):
            topLabels = class_names[top3_indices[0, i].item()]
            confidence = top3_probs[0, i].item() * 100
            print(f"{topLabels}: {confidence:.2f}%")
    
    return None

def main():
    train_loader = get_data_loader(training = True)
    # print(type(train_loader))
    # print(train_loader.dataset)
    # model = build_model()
    model = build_deeper_model()
    test_loader = get_data_loader(training = False)
    train_model(model, train_loader, criterion, 5)
    # evaluate_model(model, train_loader, criterion, show_loss = True)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    test_images = next(iter(test_loader))[0]
    predict_label(model, test_images, 1)
    


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    main()
