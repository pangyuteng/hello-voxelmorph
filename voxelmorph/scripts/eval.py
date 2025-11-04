

"""
# step 1

+ generate condor args for inference using 128^3 images.
    + using downloaded synthmorph weights
    + and updated weights 1500.h5

    + transform from tlc to rv, also rv to tlc.

"""


"""
# step 2

+ from above results:
    transform from tlc to rv, compute dice: wlung, lobes, fissure, lung-vessels, liver.
    transform from rv to tlc, compute dice: wlung, lobes, fissure, lung-vessels, liver.

"""


"""
# step 3

+ peform step 1 and 2 using 512^3 images.


"""


"""

# TODO:
+ consider retraining from scratch at 256^3 to see if dice, specifically fissur and lung-vessels, improves.
+ consider doing registration with pTVReg.

paper title... 

High Resolution Chest CT registration


"""