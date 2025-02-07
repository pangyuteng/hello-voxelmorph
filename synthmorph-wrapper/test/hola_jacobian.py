import sys
import tensorflow as tf
import SimpleITK as sitk
import voxelmorph as vxm
from voxelmorph.py.utils import jacobian_determinant
from synthmorph_wrapper import register_transform as rt


def main(fixed_file,moving_file,jdet_file,gpu_id):
    device, nb_devices = vxm.tf.utils.setup_device(gpu_id)

    add_feat_axis = not rt.MULTI_CHANNEL

    sm_fixed, fixed_affine = vxm.py.utils.load_volfile(
        fixed_file, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    sm_moving = vxm.py.utils.load_volfile(moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)


    inshape = sm_moving.shape[1:-1]
    nb_feats = sm_moving.shape[-1]

    with tf.device(device):
        # load model and predict
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(rt.MODEL_FILE,**config).register(sm_moving, sm_fixed)
        print(warp.shape)
        warp = warp.squeeze()
        print(warp.shape)
        jdet = jacobian_determinant(warp)

    print(jdet.shape)
    vxm.py.utils.save_volfile(jdet.squeeze(), jdet_file, fixed_affine)

if __name__ == "__main__":

    fixed_file = sys.argv[1]
    moving_file = sys.argv[2]
    jdet_file = sys.argv[3]
    gpu_id = sys.argv[4]
    main(fixed_file,moving_file,jdet_file,gpu_id)

"""

CUDA_VISIBLE_DEVICES=2 python hola_jacobian.py \
    moved/sm-fixed.nii.gz moved/sm-moving.nii.gz moved/jdet.nii.gz 2

"""