import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

from model import SpeechTransformer


def main():
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    
    print("====> Loading dataset <====")
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = OneHotEncoder(sparse=False).fit_transform(y_train)
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train / 1.0).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    
    model = SpeechTransformer(
        in_channels=4,
        img_height=64,
        img_width=64,
        strip_width=2,
        dim_feature=512,
        num_head=4,
        num_layers=5,
        device=device
    ).to(device)
    model.weights_init(init_type='kaiming')
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    criterion = nn.CrossEntropyLoss()
    
    print("====> Training <====")
    epochs = 200
    loss_items = []
    for epoch in range(epochs):
        pred_train = model(X_train)
        loss = criterion(pred_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print("Epoch [%d/%d], Loss: %.5f" % (epoch + 1, epochs, loss.item()))
        loss_items.append(loss.item())    
            
    print("====> Testing <====")
    pred_test = model(X_test)
    pred_test = np.argmax(pred_test.cpu().detach().numpy(), axis=1)
    conf_matrix = confusion_matrix(y_test, pred_test)
    conf_matrix = pd.DataFrame(conf_matrix, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 
                               columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    print('Accuracy Score: ', accuracy_score(y_test, pred_test))
    
    conf_matrix.to_csv("data/conf_matrix.csv")
    np.save("data/loss.npy", np.array(loss_items))


if __name__ == '__main__':
    main()
    
    
