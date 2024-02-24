import sys
import pandas as pd

csv_file = sys.argv[1]
df = pd.read_csv(csv_file).drop_duplicates()

tmpdf = df[['dest_file']]
tmpdf.to_csv('tl_raw.csv',index=False)

df = pd.read_csv(csv_file)
mylist = []
for patient_id in df.patient_id.unique():
    tmpdf = df[df.patient_id==patient_id]
    if len(tmpdf) < 2:
        continue

    # let rv be fixed?
    tmpdf = tmpdf.sort_values(['breath_hold','study_date'])
    file_list = tmpdf.dest_file.tolist()
    print(len(file_list))
    for x in file_list[1:]:
        myitem = dict(
            ref_file=file_list[0],
            dest_file=x
        )
        mylist.append(myitem)

tmpdf = pd.DataFrame(mylist)
tmpdf.to_csv('tl_paired.csv',index=False)

"""

option 1. just try registering among all images.
  tl_raw.csv
option 2. try registering using paired images.
  tl_paired.csv

"""