import os
import pandas as pd
df = pd.read_csv("tl_masked.csv")

with open("tl.csv","w") as f:
    for x in df['nifti_file']:
        if os.path.exists(x) is False:
            continue
        f.write(x+"\n")

"""
python convert_tl_to_imagelist.py
"""