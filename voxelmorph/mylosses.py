
raise NotImplementedError()
#
# D Wang, PLOSL: Population Learning Followed by One Shot Learning Pulmonary Image Registration 
# Using Tissue Volume Preserving and Vesselness Constraints
# https://pmc.ncbi.nlm.nih.gov/articles/PMC11225793
#
# Y Yin, Mass preserving nonrigid registration of CT lung images using cubic B-spline
# https://pmc.ncbi.nlm.nih.gov/articles/PMC2749644
#

# max hu 1000, min hu = -1000
# input max is 1, min is 0
# thus, hu_tissue = 0.5275 # >>> (55+1000)/(1000+1000)
from voxelmorph.py.utils import jacobian_determinant

# Sum of Squared Tissue Volume Difference
class SSTVD:
    def __init__(self,hu_tissue=0.5275,hu_air=0):
        self.hu_tissue = hu_tissue
        self.hu_air = hu_air
        pass
    def sqr_diff(self, y_true, y_pred, warp): # y_pred is moving
        tv_true = (y_true - self.hu_air)/(self.hu_tissue-self.hu_air)
        tv_pred = (y_pred - self.hu_air)/(self.hu_tissue-self.hu_air)
        '''
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])
        jdet = jacobian_determinant(warp.squeeze())
        '''
        return K.square(tv_true-jdet*tv_pred)

    def loss(self, y_true, y_pred):
        # compute loss
        loss = K.sum(self.sqr_diff(y_true, y_pred))
        return loss

