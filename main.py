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

# Print results
print("\n\n############################# Image ROCAUC Results #############################\n")
print(image_rocaucs_df.to_markdown(index=False))

print("\n\n############################# Pixel ROCAUC Results #############################\n")
print(pixel_rocaucs_df.to_markdown(index=False))

print("\n\n############################# AU PRO Results #############################\n")
print(au_pros_df.to_markdown(index=False))


# Save results
os.makedirs("results", exist_ok=True)
image_rocaucs_df.to_markdown("results/image_rocauc_results.md", index=False)
pixel_rocaucs_df.to_markdown("results/pixel_rocauc_results.md", index=False)
au_pros_df.to_markdown("results/aupro_results.md", index=False)


   


