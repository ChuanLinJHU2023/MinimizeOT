import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from utils import *



image_titles = ["2.206", "35.781", "31.671", "1.071", "38.346", "32.084","0.000","35.517","32.529"]
# image_titles = None
combine_images("output_images",layout=(3,3), image_titles=image_titles)