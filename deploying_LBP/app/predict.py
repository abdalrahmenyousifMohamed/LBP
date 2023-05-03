#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from config import CONFIG
import numpy as np


def preprocess(package: dict, input: list) -> list:
    """
    Preprocess data before running with model, for example scaling and doing one hot encoding

    :param package: dict from fastapi state including model and preocessing objects
    :param package: list of input to be proprocessed
    :return: list of proprocessed input
    """

    # scale the data based with scaler fit during training
    scaler = package['scaler']
    input = np.array(input)
    input = scaler.transform(input.reshape(1,-1))

    return input


def predict(package: dict, input: list) -> np.ndarray:
    """
    Run model and get result

    :param package: dict from fastapi state including model and preocessing objects
    :param package: list of input values
    :return: numpy array of model output
    """

    # process data
    X = preprocess(package, input)

    # run model
    model = package['model']

    X_val = torch.from_numpy(X)
    
    X_val = X_val.float()

    model.eval()
    X_val.to(CONFIG['DEVICE'])
    logits = model(X_val)
    y_pred = torch.sigmoid(logits)

    
    # with torch.no_grad():
    #     # convert input from list to Tensor
    #     X = torch.Tensor(X)

    #     # move tensor to device
    #     X = X.to(CONFIG['DEVICE'])

    #     # run model
    #     y_pred = model(X)

    # convert result to a numpy array on CPU
    # y_pred = y_pred.cpu().numpy()
    print("Y PRED IS" , y_pred)

    return y_pred.detach().numpy()