import hub
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from model import SpeechTransformer


def main():
    # preprocess the dataset
    # ds = hub.load("hub://activeloop/spoken_mnist")
    # X = ds.spectrograms.numpy()
    # y = ds.labels.numpy()
    print("====> Loading dataset <====")
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = OneHotEncoder(sparse=False).fit_transform(y_train)
    y_test = OneHotEncoder(sparse=False).fit_transform(y_test)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train / 1.0).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test / 1.0).float()
    
    model = SpeechTransformer(
        in_channels=4,
        img_height=64,
        img_width=64,
        strip_width=2,
        dim_feature=512,
        num_head=4,
        num_layers=5,
    )
    model.weights_init(init_type='kaiming')
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    criterion = nn.CrossEntropyLoss()
    print("====> Training <====")
    epochs = 10
    for epoch in range(epochs):
        pred_train = model(X_train)
        loss = criterion(pred_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch [%d/%d], Loss: %.5f" % (epoch + 1, epochs, loss.item()))


if __name__ == '__main__':
    main()
    