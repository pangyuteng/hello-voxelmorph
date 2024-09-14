
import sys
import json
from synthmorph_wrapper import register_transform as rt


json_file = sys.argv[1]

with open(json_file,'r') as f:
    content_dict = json.loads(f.read())

rt.register_transform(content_dict)

"""

docker run -it -u $(id -u):$(id -g) -v $PWD:/test -w /test pangyuteng/synthmorph-wrapper:0.1.0 bash
python test.py input.json

"""
