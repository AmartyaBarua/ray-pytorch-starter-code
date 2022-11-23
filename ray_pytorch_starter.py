import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import tune
from ray.tune.schedulers import ASHAScheduler


class Net(nn.Module):
    def __init__(self,l1=128,l2=64):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model,optimizer,train_loader,device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model,test_loader,device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        return test_loss, 100. * correct / len(test_loader.dataset)


def train_mnist(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_loc = '/home/distributedMNIST/data'

    train_dataset = datasets.MNIST(dataset_loc,
                                   download=True,
                                   train=True,
                                   transform=transform)

    # experiment with train_batch_size
    train_loader = DataLoader(train_dataset,
                              batch_size=config["train_batch_size"],
                              shuffle=False,
                              # This is mandatory to set this to False here, shuffling is done by Sampler
                              num_workers=4,
                              pin_memory=True)

    test_dataset = datasets.MNIST(dataset_loc,
                                  download=True,
                                  train=False,
                                  transform=transform)

    # experiment with test_batch_size
    test_loader = DataLoader(test_dataset,
                             batch_size=config["test_batch_size"],
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])

    for i in range(config["epoch"]):
        train(model,optimizer,train_loader,device)
        loss, acc = test(model,test_loader,device)

        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
                (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        print("Finished Training")


def main():

    search_space = {
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "momentum": tune.uniform(0.1, 0.9),
        "test_batch_size": tune.grid_search([64]),
        "train_batch_size": tune.grid_search([64]),
        "epoch": tune.grid_search([10, 15, 20]),
        "l1": tune.grid_search([128]),
        "l2": tune.grid_search([64]),
    }

    trainable_with_gpu = tune.with_resources(train_mnist,{"cpu": 6, "gpu": 3})

    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            num_samples=20,
            scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="mean_accuracy", mode="max")  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.log_dir  # Get best trial's logdir
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    best_metrics = best_result.metrics  # Get best trial's last results
    best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

    print("best_logdir: {}".format(best_logdir))
    print("best_config: {}".format(best_config))
    print("best_metric: {}".format(best_metrics))
    print(best_result_df.to_markdown())
    print("best_checkpoint: {}".format(best_checkpoint))

main()


