import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from utils import *



image_titles = ["6.439", "141.45", "124.62", "1.433", "76.132", "126.700","0.000","70.479","64.471"]
# image_titles = None
combine_images("output_images",layout=(3,3), image_titles=image_titles)