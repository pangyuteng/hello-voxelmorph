
import pandas as pd
df = pd.read_csv("tl_masked.csv")

with open("tl.csv","w") as f:
    for x in df['nifti_file']:
        f.write(x+"\n")

"""
python convert_tl_to_imagelist.py
"""