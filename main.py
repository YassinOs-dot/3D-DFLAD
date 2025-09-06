import pandas as pd
import os
from Runner import *

METHOD_NAME = "RGBFPFHFeatures"
CLASS_NAME = "cookie"

# Initialize DataFrame
image_rocaucs_df = pd.DataFrame([METHOD_NAME], columns=['Method'])
pixel_rocaucs_df = pd.DataFrame([METHOD_NAME], columns=['Method'])
au_pros_df = pd.DataFrame([METHOD_NAME], columns=['Method'])

print(f"\nRunning on class {CLASS_NAME}\n")
patchcore = Run.patchcore()  # uses fixed class and method
patchcore.fit()
image_rocauc, pixel_rocauc, au_pro = patchcore.evaluate()

# Add results to DataFrame
image_rocaucs_df[CLASS_NAME.title()] = [image_rocauc]
pixel_rocaucs_df[CLASS_NAME.title()] = [pixel_rocauc]
au_pros_df[CLASS_NAME.title()] = [au_pro]
