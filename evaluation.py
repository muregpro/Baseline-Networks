import numpy as np
from metrics import DSC, RDSC, centroid_maes, TRE, RTRE, RTs, HD95, StDJD 
from data_generator import test_generator, resize_3d_image

import voxelmorph as vxm

from tensorflow import keras

import tensorflow as tf

val_path = r'nifti_data/val'

batch_size = 1

# moving_image_shape = (81, 118, 88, 1)
# fixed_image_shape = (120, 128, 128, 1)

moving_image_shape = (64, 64, 64, 1)
fixed_image_shape = (64, 64, 64, 1)

model_save_path  =r'registration_model_trial_288'
lambda_param = 0.05

spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

registration_model = keras.models.load_model(model_save_path, custom_objects={'loss': [vxm.losses.MSE().loss, vxm.losses.Dice().loss, vxm.losses.Grad('l2').loss], 'loss_weights': [0.5, 1, lambda_param]})

def evaulation_function(moving_image, fixed_image, moving_label):
    temp_moving_label = np.zeros(np.shape(moving_image))
    temp_fixed_label = np.zeros(np.shape(fixed_image))
    _, _, ddf = registration_model.predict((moving_image, fixed_image, temp_moving_label, temp_fixed_label), verbose=0)
    moved_label = spatial_transformer([moving_label, ddf])
    return moved_label, ddf
    

all_labels_fixed_labels = []
all_labels_moved_labels = []
all_labels_moving_labels = []
all_labels_ddfs = []

for label_num in range(6):
    val_gen = test_generator(val_path, batch_size, moving_image_shape, fixed_image_shape, start_index=None, end_index=None, label_num=label_num, with_label_inputs=True)
    all_fixed_labels = []
    all_moved_labels = []
    all_moving_labels = []
    all_ddfs = []
    
    # i = 0
    while True:
        try:
            (val_inputs, val_outputs) = next(val_gen)
            moving_image, fixed_image, moving_label, fixed_label = val_inputs
            fixed_image, fixed_label, zero_phi = val_outputs
            moved_label, ddf = evaulation_function(moving_image, fixed_image, moving_label)
            # print(np.amax(fixed_label), np.amax(moving_label))
            
            all_fixed_labels.append(fixed_label)
            all_moved_labels.append(moved_label)
            all_moving_labels.append(moving_label)
            all_ddfs.append(ddf)
            # print(i)
            # i += 1
            
        except (IndexError, StopIteration) as e:
            all_labels_fixed_labels.append(np.concatenate(all_fixed_labels, axis=0))
            all_labels_moved_labels.append(np.concatenate(all_moved_labels, axis=0))
            all_labels_moving_labels.append(np.concatenate(all_moving_labels, axis=0))
            all_labels_ddfs.append(np.concatenate(all_ddfs, axis=0))
            break


all_dscs = []
all_rdscs = []
maes = []
all_hd95s = []

all_lim_hd95s = []
all_lim_maes = []

# compute metrics
for label in range(6):
    dsc = DSC(tf.convert_to_tensor(all_labels_fixed_labels[label], dtype=tf.double), tf.convert_to_tensor(all_labels_moved_labels[label], dtype=tf.double))
    rdsc = RDSC(tf.convert_to_tensor(all_labels_fixed_labels[label], dtype=tf.double), tf.convert_to_tensor(all_labels_moved_labels[label], dtype=tf.double))
    mae = centroid_maes(tf.convert_to_tensor(all_labels_fixed_labels[label], dtype=tf.double), tf.convert_to_tensor(all_labels_moved_labels[label], dtype=tf.double))
    hd95 = HD95(tf.convert_to_tensor(all_labels_fixed_labels[label], dtype=tf.double), tf.convert_to_tensor(all_labels_moved_labels[label], dtype=tf.double))
    
    lim_hd95 = HD95(tf.convert_to_tensor(all_labels_fixed_labels[label], dtype=tf.double), tf.convert_to_tensor(all_labels_moving_labels[label], dtype=tf.double))
    lim_mae = centroid_maes(tf.convert_to_tensor(all_labels_fixed_labels[label], dtype=tf.double), tf.convert_to_tensor(all_labels_moving_labels[label], dtype=tf.double))
    
    all_dscs.append(dsc)
    all_rdscs.append(rdsc)
    maes.append(mae)
    all_hd95s.append(hd95)
    
    all_lim_hd95s.append(lim_hd95)
    all_lim_maes.append(lim_mae)
    

fin_DSC = np.mean(all_dscs)
fin_RDSC = np.mean(all_rdscs)


fin_HD95 = np.mean(all_hd95s)
fin_lim_HD95 = np.mean(all_lim_hd95s)

maes = [np.expand_dims(elem, -1) for elem in maes]
all_maes_array = np.concatenate(maes, axis=-1)
idx = np.argwhere(np.all(all_maes_array[..., :] == 0, axis=0))
all_maes_array = np.delete(all_maes_array, idx, axis=1)

lim_maes = [np.expand_dims(elem, -1) for elem in all_lim_maes]
all_lim_maes_array = np.concatenate(lim_maes, axis=-1)
idx = np.argwhere(np.all(all_lim_maes_array[..., :] == 0, axis=0))
all_lim_maes_array = np.delete(all_lim_maes_array, idx, axis=1)

fin_TRE = TRE(all_maes_array)
fin_RTRE = RTRE(all_maes_array)
fin_RTs = RTs(all_maes_array)

fin_lim_TRE = TRE(all_lim_maes_array)
fin_lim_RTRE = RTRE(all_lim_maes_array)
fin_lim_RTs = RTs(all_lim_maes_array)


# stdjds = []
# for ddf in all_ddfs:
#     stdjd = StDJD(ddf[0])
#     stdjds.append(stdjd)

# fin_StDJD = np.mean(stdjds)



print(f'\nALL METRICS:\n\n'
      f'DSC: {fin_DSC}\n',
      f'RDSC: {fin_RDSC}\n',
      f'HD95: {fin_HD95/fin_lim_HD95}\n',
      f'TRE: {fin_TRE/fin_lim_TRE}\n',
      f'RTRE: {fin_RTRE/fin_lim_RTRE}\n',
      f'RTs: {fin_RTs/fin_lim_RTs}\n',
      # f'StDJD: {fin_StDJD}\n'
      )

def score_calculator(fin_DSC, fin_RDSC, fin_HD95, fin_lim_HD95, fin_TRE, fin_lim_TRE, fin_RTRE, fin_lim_RTRE, fin_RTs, fin_lim_RTs):
    score = 0.2*fin_DSC + 0.1*(fin_RDSC) + 0.3*(1-(fin_TRE/fin_lim_TRE)) + 0.1*(1-(fin_RTRE/fin_lim_RTRE)) + 0.1*(1-(fin_RTs/fin_lim_RTs)) + 0.2*(1-(fin_HD95/fin_lim_HD95))
    return score

score = score_calculator(fin_DSC, fin_RDSC, fin_HD95, fin_lim_HD95, fin_TRE, fin_lim_TRE, fin_RTRE, fin_lim_RTRE, fin_RTs, fin_lim_RTs)

print(f'\nFinal Score: {score}')














# class Evaluation():
#     def __init__(self):
        
#         val_path = r'/home/s-sd/Desktop/mu_reg_miccai_challenge/nifti_data/holdout'
        
#         batch_size = 1
        
#         # moving_image_shape = (81, 118, 88, 1)
#         # fixed_image_shape = (120, 128, 128, 1)
        
#         moving_image_shape = (64, 64, 64, 1)
#         fixed_image_shape = (64, 64, 64, 1)
        
#         model_save_path  =r'/home/s-sd/Desktop/mu_reg_miccai_challenge/ckpts_docker/localnet_model_checkpoints/registration_model_trial_288'
#         lambda_param = 0.05
        
#         spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')
        
#         registration_model = keras.models.load_model(model_save_path, custom_objects={'loss': [vxm.losses.MSE().loss, vxm.losses.Dice().loss, vxm.losses.Grad('l2').loss], 'loss_weights': [0.5, 1, lambda_param]})
