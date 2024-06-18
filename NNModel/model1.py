import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style()

class NNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential( nn.Linear(dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 64), 
                                 nn.ReLU(), 
                                 nn.Linear(64, 2),
                                 nn.Softmax()
        )
    def forward(self, Xtr):
        return self.fc(Xtr)

if __name__ == '__main__':
    dbg = 'off'
    np.random.seed(1919)
    M = 1024
    X = torch.tensor(np.random.normal(size=(M, 1)), dtype = torch.float32)
    Y = torch.tensor([[1,0] if x > 0 else [0,1] for x in X], dtype = torch.float32)
    Xtr, Xtest, Ytr, Ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1919)
    # training.
    nn_model = NNetwork(Xtr.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(nn_model.parameters(), lr=0.01)
    epochs = 10
    # mini-batches.
    batch_size = 32
    # IMP -> https://stackoverflow.com/questions/67683406/difference-between-dataset-and-tensordataset-in-pytorch
    train_dataset = TensorDataset(Xtr, Ytr)
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
    plot_epoch, plot_loss = [], []
    for _ in tqdm(range(epochs)):
        total_loss, cnt_loss = 0, 0
        for xb, yb in train_loader:
            yp = nn_model(xb)
            loss = criterion(yp, yb)
            total_loss += loss.item(); cnt_loss += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        plot_epoch.append(_+1)
        plot_loss.append(total_loss/cnt_loss)
    # plot training loss. 
    plt.plot(plot_epoch, plot_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss v/s Epoch.')
    if dbg =="on": plt.show()
    # testing.
    with torch.no_grad():
        YtestP = nn_model(Xtest)
        acc = (YtestP.round() == Ytest).float().mean()
        print(f'\nTest Accuracy: {round(acc.item()*100, 2)}%\n')