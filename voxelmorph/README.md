
+ build container
(torch+tf+lib requirements based on # https://colab.research.google.com/drive/14s2h0j_Aoncp587vmpjsQBe6PQTxyVP5#scrollTo=8hBjrxrAwmaw)

```

bash build_and_push.sh

```

+ `TODO` sample scripts see `scripts`.


### TODO

[x] add sample registration script for tlc & rv

[ ] add sample training and evaluation scripts for a. chect ct scans, b. do we need to fine tung using paired images? same subjects, tlc&rv with multipled visits.


### containers tags

+ torch==2.8.0+cu129 & tf_keras==2.19.0 voxelmorph `pangyuteng/voxelmorph:0.1.2`

+ tensorflow==2.11.0 voxelmorph: `pangyuteng/voxelmorph:0.1.1`

