import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from wgu3_gan import gan
#%%

def run(self):
    print("Backtesting complete")
    return True

# TODO Figure out why wgu3_gan isn't recognised as package
#fixme