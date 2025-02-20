```

using a blen of below Par0011 and Par0054 config

ref
https://github.com/SuperElastix/ElastixModelZoo/tree/master/models/Par0011
https://github.com/SuperElastix/ElastixModelZoo/tree/master/models/Par0054

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2749644



+ also can be registered with mask, but results with mask is not better

elastix -f $FIXED_NIFTI -m $MOVING_NIFTI \
    -fMask fixed_mask.nii.gz -mMask moving_mask.nii.gz \ 
    -out $AFFINE_DIR \
    -p param/affine.txt


```