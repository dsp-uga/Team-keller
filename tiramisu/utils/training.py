import os
import sys
import math
import string
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'


def save_weights(model, epoch, loss, err):
    """
    save the weight at specific epochs.
    Parameters
    ----------
    model: Model
        The model with the weight that has been trained.
    epoch: int
        The current epoch.
    loss: float
        The current loss.
    error: float
        The current error.
    """
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
        'startEpoch': epoch,
        'loss': loss,
        'error': err,
        'state_dict': model.state_dict()
    }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH + 'latest.th')


def load_weights(model, fpath):
    """
    load the weight from saved files.
    Parameters
    ----------
    model: Model
        The model with the weight that is going to be loaded.
    fpath: string
        The path of the weight file.
    """
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch - 1, weights['loss'], weights['error']))
    return startEpoch


def get_predictions(output_batch):
    """
    get prediction from a batch of data.
    Parameters
    ----------
    output_batch: Tensor
        the batch of data that are feeded into the model for predictions.
    """
    bs, c, h, w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs, h, w)
    return indices


def error(preds, targets):
    """
    get the error given the predictions and the ground truths.
    Parameters
    ----------
    preds: Tensor
        the predictions given the inputs.
    targets: Tensor
        the ground truth of the inputs.
    """
    assert preds.size() == targets.size()
    bs, h, w = preds.size()
    n_pixels = bs * h * w
    incorrect = preds.ne(targets).cpu().sum().type(torch.FloatTensor)
    err = incorrect / n_pixels
    return round(err.item(), 5)


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(
                0, 1).transpose(
                0, 2).squeeze().contiguous().long()
        return label


def train(model, trn_loader, optimizer, criterion, epoch):
    """
    Model training function.
    Parameters
    ----------
    model: Model
        The model with the weight that is going to be trained.
    trn_loader: data loader
        The loader for the training data.
    optimizer: Optimizer
        The optimizer of the training.
    criterion: Loss function
        The loss function of the training.
    epoch: int
        The training epoch.
    """
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        inputs = data[0].cuda()
        targets = data[1].cuda()

        optimizer.zero_grad()
        output = model(inputs.float())
        loss = criterion(output, LabelToLongTensor()(targets))
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error


def test(model, test_loader, criterion, epoch=1):
    """
    Model testing function.
    Parameters
    ----------
    model: Model
        The model with the weight that is going to be tested.
    test_loader: data loader
        The loader for the testing data.
    criterion: Loss function
        The loss function of the testing.
    epoch: int
        The testing epoch, default set as 1.
    """
    model.eval()
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = get_predictions(output)
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """
    Sets the learning rate to the initially configured `lr` decayed by `decay` every `n_epochs
    Parameters
    ----------
    lr: float
        The current learning rate.
    decay: float
        The decay rate.
    cur_epoch: int
        The current epoch.
    n_epochs: int
        The learning rate adjust frequency by epochs.
    """
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weights_init(m):
    """
    initializing the weights for convolutional layers.
    Parameters
    ----------
    m: layer
        the convolutional layer to be initialized.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def predict(model, input_loader, n_batches=1):
    """
    Model predicts the input.
    Parameters
    ----------
    model: Model
        The model with the weight that is going to perform the prediction.
    input_loader: data loader
        The loader for the input.
    n_batches: int
        the batches of the inputs, default set as 1.
    """
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input, target, pred])
    return predictions


def view_sample_predictions(model, loader, n):
    """
    Model predicts the input with an visualized output.
    Parameters
    ----------
    model: Model
        The model with the weight that is going to perform the prediction.
    loader: data loader
        The loader for the input.
    n: int
        the visualized images range. First n images are visualized.
    """
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    label = Variable(targets.cuda())
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])
