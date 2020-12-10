import torch
import numpy as np
from tqdm import trange, tqdm
import torch.nn.functional as F
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import  time

from model import Predictor
from dataset import PickupDS



def step(net, x, y, loss, device, eps=0.0001):
    x, y = x.to(device), y.to(device)
    y_pred = net(x)
    if loss == 'mse':
        return F.mse_loss(y_pred, y), y_pred, y
    return torch.abs((y_pred - y) / (y + eps)).mean(), y_pred, y


def main(args):
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Predictor(cell_type=args['cell'], hidden_size=args['hidden'], dropout=args['drop'], num_layers=args['layers'])
    net = net.to(device)
    net.train()
    if args['optim'] == 'sgd':
        optimizer = SGD(net.parameters(), lr=0.001)
    elif args['optim'] == 'adam':
        optimizer = Adam(net.parameters(), lr=3e-4)
    else:
        optimizer = RMSprop(net.parameters(), lr=0.01)
    dataset = PickupDS()
    train_indices = np.arange(int(len(dataset) * 0.8))
    train_data_loader = DataLoader(dataset, batch_size=args['bs'],
                                   sampler=SubsetRandomSampler(train_indices), drop_last=True)
    val_indices = np.arange(int(len(dataset) * 0.2), len(dataset))
    val_data_loader = DataLoader(Subset(dataset, val_indices), batch_size=args['bs'], drop_last=False)
    train_losses = []
    val_losses = []
    for _ in trange(args['epochs']):
        current_train_losses = []
        current_val_losses = []
        net.train()
        y_preds = []
        y_targets = []
        for x, y in tqdm(train_data_loader):

            loss, y_pred, y_target = step(net, x, y, args['loss'], device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_train_losses.append(loss.item())
        net.eval()

        with torch.no_grad():
            for x, y in tqdm(val_data_loader):
                loss, y_pred, y_target = step(net, x, y, args['loss'], device)
                current_val_losses.append(loss.item())
                y_preds.append(y_pred.cpu().numpy())
                y_targets.append(y_target.cpu().numpy())
        train_losses.append(sum(current_train_losses) / len(current_train_losses))
        val_losses.append(sum(current_val_losses) / len(current_val_losses))
    print('{} seconds'.format(time.time() - t0))
    plt.xlabel('epochs')
    plt.ylabel('error')
    # plt.legend()
    plt.plot(val_losses, label='validation' , color='r')
    plt.plot(train_losses, label='train', color='b')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    y_preds = np.concatenate(y_preds, axis=0)
    y_targets = np.concatenate(y_targets, axis=0)

    acc = 100 * ((np.sum((np.abs(np.array(y_targets) - np.array(y_preds))) / np.abs((np.array(y_targets))))))/ len(np.array(y_targets))
    # print(acc)
    print('A')
    print((1-(np.sum(np.square((np.array(y_targets[:, 0]) - np.array(y_preds[:, 0]))))) / len(y_targets)) * 100)




    for i in range(4):
        plt.xlabel('time')
        plt.ylabel('pickup')
        plt.subplot(2, 2, i + 1)
        plt.plot(y_preds[:, i], label='prediction', color='r')
        plt.plot(y_targets[:, i], label='real', color='b')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--cell', choices=['rnn', 'gru', 'lstm'], default='lstm')
    argument_parser.add_argument('--hidden', type=int, default=32)
    argument_parser.add_argument('--drop', type=float, default=0.7)
    argument_parser.add_argument('--layers', type=int, default=1)
    argument_parser.add_argument('--loss', choices=['mse', 'mape'], default='mse')
    argument_parser.add_argument('--optim', choices=['sgd', 'adam', 'rms'], default='adam')
    argument_parser.add_argument('--bs', type=int, default=32, help='batch size')
    argument_parser.add_argument('--epochs', type=int, default=20)
    return vars(argument_parser.parse_args())


if __name__ == '__main__':
    main(parse_args())