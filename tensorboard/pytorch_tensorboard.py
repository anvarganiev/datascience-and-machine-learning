import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

# костыль для write.add_embedding()
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        # same convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


model = CNN()

x = torch.randn(64, 1, 28, 28)
print(model(x).shape)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channels = 1
num_classes = 10
# learning_rate = 0.001
# batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

batch_sizes = [256]
learning_rates = [0.001]
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for batch_size in batch_sizes:
    for learning_rate in learning_rates:

        step = 0

        # initialize network
        model = CNN(in_channels=in_channels, num_classes=num_classes)
        model.to(device)
        model.train()
        # train Network
        writer = SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}')
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

        # Visualize model in TensorBoard
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(device))
        writer.close()

        for epoch in range(num_epochs):
            losses = []
            accuracies = []

            for batch_idx, (data, targets) in enumerate(train_loader):
                # get data to GPU if possible
                data = data.to(device)
                targets = targets.to(device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # calculate 'running' training accuracy
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct) / float(data.shape[0])
                accuracies.append(running_train_acc)

                features = data.reshape(data.shape[0], -1)
                class_labels = [classes[label] for label in predictions]

                # visualizing data and weights of fc1 for each batch
                img_grid = torchvision.utils.make_grid(data)
                writer.add_image('mnist_images', img_grid)
                writer.add_histogram('fc1', model.fc1.weight)



                # data shape is [batch_size, 1, 28, 28]
                # plot things to tensorboard
                writer.add_scalar('Training Loss', loss, global_step=step)
                writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)

                # PCA for images
                if batch_idx == 100:
                    writer.add_embedding(features, metadata=class_labels, label_img=data,
                                         global_step=batch_idx)

                step += 1

            writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                               {'mean_accuracy': sum(accuracies) / len(accuracies),
                                'loss': sum(losses) / len(losses)})

            print(f'Mean loss this epoch was {sum(losses) / len(losses)}')
