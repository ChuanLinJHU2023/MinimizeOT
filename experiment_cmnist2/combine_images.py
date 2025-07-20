import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from utils import *



image_titles = ["1.046", "10.452", "9.437", "0.695", "11.180", "9.480","0.000","10.374","9.648"]
# image_titles = None
combine_images("output_images",layout=(3,3), image_titles=image_titles)