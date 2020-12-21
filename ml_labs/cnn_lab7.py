import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.02
BATCH_SIZE = 64
EPOCH = 5


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.f1 = torch.nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.f2 = torch.nn.Linear(in_features=120, out_features=84)
        self.f3 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def calc_all_classes():
    s = set()
    for X_batch, y_batch in train_loader:
        for i in y_batch:
            s.add(i.item())
    res = {item: ind for ind, item in enumerate(list(s))}
    # print(res)
    return res


def train():
    net.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = net(X_batch)
        loss_value = loss_function(y_pred, y_batch)
        loss_value.backward()
        optimizer.step()


def test():
    correct = 0
    net.eval()
    error_matrix = [[0 for _ in range(len(classes))] for _ in range(len(classes))]
    img_matrix = [[None for _ in range(len(classes))] for _ in range(len(classes))]
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = net(X_batch)

            pred = y_pred.data.max(1, keepdim=True)[1]
            y_real = y_batch.data.view_as(pred)

            for i in range(len(pred)):
                i_pred = pred[i].item()
                i_real = y_real[i].item()
                if error_matrix[classes[i_pred]][classes[i_real]] == 0:
                    img = X_batch.data[i][0]
                    img_matrix[classes[i_real]][classes[i_pred]] = img
                    # plt.imshow(img, interpolation='none')
                    # plt.title(f"Real: {i_real}, Predict: {i_pred} ")
                    # plt.savefig(f"lab7_files/{i_real}_{i_pred}.png")
                    # plt.show()

                error_matrix[classes[i_pred]][classes[i_real]] += 1
                if i_real == i_pred:
                    correct += 1

    return error_matrix, correct / len(test_loader.dataset), img_matrix


def print_matrix(matrix):
    for m in matrix:
        print(m)


def draw_matrix(matrix):
    plt.figure()
    for i in range(len(matrix) ** 2):
        j = i % len(matrix)
        ii = i // len(matrix)
        plt.subplot(len(matrix), len(matrix), i + 1)
        if matrix[ii][j] is not None:
            plt.imshow(matrix[ii][j], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()


def run():
    for epoch in range(EPOCH):
        train()
        print(f"epoch {epoch}")
        # epoch_res = test()
        # print(f'EPOCH ACCURACY: {epoch_res}')
    error_matrix, acc, img_matrix = test()
    print(np.array(error_matrix))
    draw_matrix(img_matrix)
    print(f"Accuracy: {acc}")


net = Net()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./', train=True, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=BATCH_SIZE, shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./', train=False, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=BATCH_SIZE, shuffle=True)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('./', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('./', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE, shuffle=True)

# for X_batch, y_batch in test_loader:
#     img = X_batch.data[0][0]
#     plot.imshow(img, cmap='gray', interpolation='none')
#     plot.show()

classes = calc_all_classes()
run()
