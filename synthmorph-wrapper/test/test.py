import os
import sys
import json
import SimpleITK as sitk
from synthmorph_wrapper import register_transform as rt

# json_file = sys.argv[1]
def dcm2nifti(folder_path):
    nifti_file = os.path.join(folder_path,"image.nii.gz")
    if not os.path.exists(nifti_file):
        dcm_list = [(int(x.split("-")[0]),x) for x in os.listdir(folder_path) if x.endswith(".dcm")]
        dcm_list = [os.path.join(folder_path,x[1]) for x in sorted(dcm_list,key=lambda x:x[0])]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dcm_list)
        img_obj = reader.Execute()
        sitk.WriteImage(img_obj,nifti_file)

dcm2nifti("image-1")
dcm2nifti("image-2")

json_file = "input.json"

with open(json_file,'r') as f:
    content_dict = json.loads(f.read())

rt.register_transform(content_dict)

"""

docker run -it -u $(id -u):$(id -g) -v $PWD:/test -w /test pangyuteng/synthmorph-wrapper:0.1.0 bash
python test.py input.json

"""

