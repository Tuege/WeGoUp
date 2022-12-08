import multiprocessing as mp
import os
import sys
import time
import importlib

from DCGAN import dcgan as gan
import DCGAN


def count():
    for n in range(10):
        print(n)
        for i in range(200000000):
            pass


def train_gan():
    """gan.trainer(params)
    gan.optimiser(params)
    gan.hyperparameter_optimiser(params)
    gan.train(dataset, hyperparameter_optimisation=True)"""
    gan.train()


if __name__ == '__main__':
    mp.freeze_support()

    gan_thread = mp.Process(target=gan.train)
    other_thread = mp.Process(target=count)
    # gan_thread.start()
    # other_thread.start()



