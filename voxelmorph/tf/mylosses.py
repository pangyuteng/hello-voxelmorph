#
# D Wang, PLOSL: Population Learning Followed by One Shot Learning Pulmonary Image Registration 
# Using Tissue Volume Preserving and Vesselness Constraints
# https://pmc.ncbi.nlm.nih.gov/articles/PMC11225793
#
# Y Yin, Mass preserving nonrigid registration of CT lung images using cubic B-spline
# https://pmc.ncbi.nlm.nih.gov/articles/PMC2749644
#
# https://pubmed.ncbi.nlm.nih.gov/27076353/

# max hu 1000, min hu = -1000
# input max is 1, min is 0
# thus, hu_tissue = 0.5275 # >>> (55+1000)/(1000+1000)

# min_val=-1000,max_val=1000,out_min_val=0.0,out_max_val=1.0
# Sum of Squared Tissue Volume Difference, although D Wang calls it SSID
import tensorflow.keras.backend as K

class SSIDLoss:
    def __init__(self,hu_tissue=0.5275,hu_air=0):
        self.hu_tissue = hu_tissue
        self.hu_air = hu_air
        pass
    def sqr_diff(self, y_true, y_pred): # y_pred is moving
        tv_true = (y_true - self.hu_air)/(self.hu_tissue-self.hu_air)
        tv_pred = (y_pred - self.hu_air)/(self.hu_tissue-self.hu_air)
        return K.square(tv_true-tv_pred)

    def loss(self, y_true, y_pred):
        # compute loss # TODO: if we have 2 channels,,, pick first channel
        loss = K.sum(self.sqr_diff(y_true, y_pred))
        # loss = K.sum(self.sqr_diff(y_true[:,0,:,:,:], y_pred[:,0,:,:,:]))
        return loss

class SSIDSSVMDLoss:
    def __init__(self,alpha=1.0):
        self.alpha = alpha
        self.ssid = SSIDLoss()
    def sqr_diff(self, y_true, y_pred): # y_pred is moving
        return K.square(y_true-y_pred)

    def loss(self, y_true, y_pred):
        # compute loss # TODO: pick second channel
        raise NotImplementedError()
        ssvmd_loss = K.sum(self.sqr_diff(y_true[:,1,:,:,:], y_pred[:,1,:,:,:]))
        ssid_loss =self.ssid.loss(y_true, y_pred)
        return ssid_loss + ssvmd_loss*self.alpha