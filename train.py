#! /usr/bin/env python3

from model import *
import numpy as np
import os
import sys
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from matplotlib import pyplot

EPOCHS = 20
REPS = 3
LR = 1e-1

def train_inequality():
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for rep in range(REPS):

        print('\n-- Starting rep #%d --\n'%(rep))

        for activation in NonLinearity:

            print('Training %s:'%(activation.name))

            model = InequalityModel(activation)
            sgd = SGD(model.parameters(), lr=LR)
            loss_fn = CrossEntropyLoss()
            all_epoch = EPOCHS
            
            accuracies = [0]*all_epoch

            for current_epoch in range(all_epoch):
                model.train()
                for idx, (train_x, train_label) in enumerate(train_loader):
                    train_x = train_x
                    train_label = train_label
                    sgd.zero_grad()
                    predict_y = model(train_x.float())
                    loss = loss_fn(predict_y, train_label.long())
                    loss.backward()
                    sgd.step()

                all_correct_num = 0
                all_sample_num = 0

                model.eval()
                for idx, (test_x, test_label) in enumerate(test_loader):
                    test_x = test_x
                    test_label = test_label
                    predict_y = model(test_x.float()).detach()
                    predict_y = torch.argmax(predict_y, dim=-1)
                    current_correct_num = predict_y == test_label
                    all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
                    all_sample_num += current_correct_num.shape[0]

                acc = all_correct_num / all_sample_num
                accuracies[current_epoch] = acc
                print('\tFinished epoch %s (%.2f)'%(current_epoch+1, acc))

            print('Finished training\n')

            pyplot.xlabel("Epoch")
            pyplot.ylabel("Accuracy")
            pyplot.plot([x+1 for x in list(range(all_epoch))], accuracies, label=activation.name)

        pyplot.legend(loc='best')
        if not os.path.isdir("plots"):
            os.mkdir("plots")
        pyplot.savefig('plots/inequality_%d.png'%(rep), bbox_inches='tight')
        pyplot.clf()


def train_recurrent():
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for rep in range(REPS):

        print('\n-- Starting rep #%d --\n'%(rep))

        for recurrence in RecType:

            print('Training %s:'%(recurrence.name))

            model = RecurrentModel(recurrence)
            sgd = SGD(model.parameters(), lr=LR)
            loss_fn = CrossEntropyLoss()
            all_epoch = EPOCHS
            accuracies = [0]*all_epoch

            for current_epoch in range(all_epoch):
                model.train()
                for idx, (train_x, train_label) in enumerate(train_loader):
                    train_x = train_x
                    train_label = train_label
                    sgd.zero_grad()
                    predict_y = model(train_x.float())
                    loss = loss_fn(predict_y, train_label.long())
                    loss.backward()
                    sgd.step()

                all_correct_num = 0
                all_sample_num = 0

                model.eval()
                for idx, (test_x, test_label) in enumerate(test_loader):
                    test_x = test_x
                    test_label = test_label
                    predict_y = model(test_x.float()).detach()
                    predict_y =torch.argmax(predict_y, dim=-1)
                    current_correct_num = predict_y == test_label
                    all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
                    all_sample_num += current_correct_num.shape[0]

                acc = all_correct_num / all_sample_num
                accuracies[current_epoch] = acc
                print('\tFinished epoch %s (%.2f)'%(current_epoch+1, acc))

            print('Finished training\n')

            pyplot.xlabel("Epoch")
            pyplot.ylabel("Accuracy")
            pyplot.plot([x+1 for x in list(range(all_epoch))]. accuracies, label=recurrence.name)

        pyplot.legend(loc='best')
        pyplot.title('Repetition #%d'%(rep))
        if not os.path.isdir("plots"):
            os.mkdir("plots")
        pyplot.savefig('plots/recurrence_%d.png'%(rep), bbox_inches='tight')
        pyplot.clf()

def train_original():
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = Model()
    sgd = SGD(model.parameters(), lr=LR)
    loss_fn = CrossEntropyLoss()
    all_epoch = EPOCHS
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x
            train_label = train_label
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x
            test_label = test_label
            predict_y = model(test_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.3f}'.format(acc), flush=True)

        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))

    print("Model finished training")

def main():
    if (len(sys.argv) < 2):
        print("Not enough arguments")
        return

    test_num = eval(sys.argv[1])
    if (test_num == 0):
        train_original()
    elif (test_num == 1):
        pass
    elif (test_num == 2):
        train_inequality()
    elif (test_num == 3):
        pass
    elif (test_num == 4):
        pass
    elif (test_num == 5):
        train_recurrent()

if __name__ == '__main__':
    main()
