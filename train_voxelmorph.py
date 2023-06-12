import os

os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import voxelmorph as vxm

import numpy as np
import tensorflow as tf

from tensorflow import keras

from model import get_model

from data_generator import train_generator, test_generator

import matplotlib.pyplot as plt

# =============================================================================
# Build the backbone model
# =============================================================================

moving_image_shape = (64, 64, 64, 1)
fixed_image_shape = (64, 64, 64, 1)

model = get_model(moving_image_shape, fixed_image_shape, with_label_inputs=False)

print('\nBackbone model inputs and outputs:')

print('    input shape: ', ', '.join([str(t.shape) for t in model.inputs]))
print('    output shape:', ', '.join([str(t.shape) for t in model.outputs]))

# =============================================================================
# Build the registration network
# =============================================================================

# build transformer layer
spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

# extract the moving image
moving_image = model.input[0]

# extract ddf
ddf = model.output

# warp the moving image with the transformer using network-predicted ddf
moved_image = spatial_transformer([moving_image, ddf])

outputs = [moved_image, ddf]

registration_model = keras.Model(inputs=model.inputs, outputs=outputs)

print('\nRegistration network inputs and outputs:')

print('    input shape: ', ', '.join([str(t.shape) for t in registration_model.inputs]))
print('    output shape:', ', '.join([str(t.shape) for t in registration_model.outputs]))

losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
lambda_param = 0.05
loss_weights = [1, lambda_param]

registration_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# =============================================================================
# Training loop
# =============================================================================

f_path = r'nifti_data/train'

val_path = r'nifti_data/val'

model_save_path = r'voxelmorph_model_checkpoints'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

batch_size = 32

train_gen = train_generator(f_path, batch_size, moving_image_shape, fixed_image_shape, with_label_inputs=False)

num_trials = 1024

val_dice = []

# registration_model = keras.models.load_model(os.path.join(model_save_path, 'registration_model_trial_328'), custom_objects={'loss': [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss], 'loss_weights': [1, lambda_param]})

for trial in range(0, num_trials):
    print(f'\nTrial {trial} / {num_trials-1}:')
    
    hist = registration_model.fit(train_gen, epochs=1, steps_per_epoch=32, verbose=1);
    
    dice_scores = []
    for label_num in range(6):
        val_gen = test_generator(f_path, 4, moving_image_shape, fixed_image_shape, start_index=None, end_index=None, label_num=label_num, with_label_inputs=True)
        while True:
            try:
                (val_inputs, val_outputs) = next(val_gen)
                moving_images_val, fixed_images_val, moving_labels_val, fixed_labels_val = val_inputs
                fixed_images_val, fixed_labels_val, zero_phis_val = val_outputs
                _, ddf_val = registration_model.predict((moving_images_val, fixed_images_val), verbose=0)
                
                moved_labels_val = spatial_transformer([moving_labels_val, ddf_val])
                moved_images_val = spatial_transformer([moving_images_val, ddf_val])
                
                dice_score = np.array(-1.0 * vxm.losses.Dice().loss(tf.convert_to_tensor(moved_labels_val, dtype='float32'), tf.convert_to_tensor(fixed_labels_val, dtype='float32')))
                dice_scores.append(dice_score)
            except (IndexError, StopIteration) as e:
                break
    val_dice.append(np.mean(dice_scores))
    plt.plot(val_dice, 'r')
    plt.xlabel('Trials')
    plt.ylabel('Dice')
    plt.savefig(r'voxelmorph_val_dice_1.png')
    print('    Validation Dice: ', np.mean(dice_scores))
    if trial % 8 == 0:
        registration_model.save(os.path.join(model_save_path, f'registration_model_trial_{trial}'))
        print('Model saved!')
