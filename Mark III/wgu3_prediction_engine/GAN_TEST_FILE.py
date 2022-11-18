import multiprocessing as mp
import os
import sys
import time

from DCGAN import dcgan as gan


def count():
    for n in range(10):
        print(n)
        for i in range(200000000):
            pass


if __name__ == '__main__':
    mp.freeze_support()

    gan_thread = mp.Process(target=gan.train)
    other_thread = mp.Process(target=count)
    gan_thread.start()
    other_thread.start()



