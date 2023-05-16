import os
import numpy as np

from skimage.transform import resize

import nibabel as nib

def resize_3d_image(image, shape):
    resized_image = resize(image, output_shape=shape)
    if np.amax(resized_image) == np.amin(resized_image):
        normalised_image = resized_image
    else:
        normalised_image = (resized_image-np.amin(resized_image))/(np.amax(resized_image)-np.amin(resized_image))
    return normalised_image

def train_generator(f_path, batch_size, moving_image_shape, fixed_image_shape, with_label_inputs=True):
    moving_images_path = os.path.join(f_path, 'us_images')
    fixed_images_path = os.path.join(f_path, 'mr_images')
    
    if with_label_inputs:
        moving_labels_path = os.path.join(f_path, 'us_labels')
        fixed_labels_path = os.path.join(f_path, 'mr_labels')
    
    all_names = np.array(os.listdir(fixed_images_path))
    
    while True:
    
        batch_names = all_names[np.random.permutation(len(all_names))[:batch_size]]
        
        moving_images_batch = np.zeros((batch_size, *moving_image_shape))
        fixed_images_batch = np.zeros((batch_size, *fixed_image_shape))
        
        if with_label_inputs:
            moving_labels_batch = np.zeros((batch_size, *moving_image_shape))
            fixed_labels_batch = np.zeros((batch_size, *fixed_image_shape))
        
        for i, f_name in enumerate(batch_names):
            moving_image = nib.load(os.path.join(moving_images_path, f_name)).get_fdata()
            fixed_image = nib.load(os.path.join(fixed_images_path, f_name)).get_fdata()

            if with_label_inputs:
                moving_label = nib.load(os.path.join(moving_labels_path, f_name)).get_fdata()
                fixed_label = nib.load(os.path.join(fixed_labels_path, f_name)).get_fdata()
            
                label_to_select = np.random.randint(6) #pick one label randomly for training
            
            moving_images_batch[i] = resize_3d_image(moving_image, moving_image_shape)
            fixed_images_batch[i] = resize_3d_image(fixed_image, fixed_image_shape)

            if with_label_inputs:
                moving_labels_batch[i] = resize_3d_image(moving_label[:, :, :, label_to_select], moving_image_shape)
                fixed_labels_batch[i] = resize_3d_image(fixed_label[:, :, :, label_to_select], fixed_image_shape)
    
        zero_phis = np.zeros([batch_size, *moving_image_shape[:-1], 3])
        
        if with_label_inputs:
            inputs = [moving_images_batch, fixed_images_batch, moving_labels_batch, fixed_labels_batch]
            outputs = [fixed_images_batch, fixed_labels_batch, zero_phis]
        else:
            inputs = [moving_images_batch, fixed_images_batch]
            outputs = [fixed_images_batch, zero_phis]
        
        yield (inputs, outputs)

def test_generator(f_path, batch_size, moving_image_shape, fixed_image_shape, start_index, end_index, label_num, with_label_inputs=True):
    moving_images_path = os.path.join(f_path, 'us_images')
    fixed_images_path = os.path.join(f_path, 'mr_images')
    
    if with_label_inputs:
        moving_labels_path = os.path.join(f_path, 'us_labels')
        fixed_labels_path = os.path.join(f_path, 'mr_labels')
    
    all_names = np.array(os.listdir(fixed_images_path))[start_index: end_index]
    
    if start_index and end_index is not None:
        n_steps = int(np.floor((end_index - start_index) / batch_size))
    else:
        start_index = 0
        end_index = len(all_names)
        n_steps =int( np.floor((end_index - start_index) / batch_size))
    
    for step in range(n_steps):
    
        batch_names = all_names[start_index: start_index + (batch_size * (step+1))]
        
        moving_images_batch = np.zeros((batch_size, *moving_image_shape))
        fixed_images_batch = np.zeros((batch_size, *fixed_image_shape))

        if with_label_inputs:
            moving_labels_batch = np.zeros((batch_size, *moving_image_shape))
            fixed_labels_batch = np.zeros((batch_size, *fixed_image_shape))
        
        for i, f_name in enumerate(batch_names):
            moving_image = nib.load(os.path.join(moving_images_path, f_name)).get_fdata()
            fixed_image = nib.load(os.path.join(fixed_images_path, f_name)).get_fdata()
            
            if with_label_inputs:
                moving_label = nib.load(os.path.join(moving_labels_path, f_name)).get_fdata() # if label not available, just pass zeros
                fixed_label = nib.load(os.path.join(fixed_labels_path, f_name)).get_fdata() # if label not available, just pass zeros
            
                label_to_select = label_num #pick one label randomly for training
            
            moving_images_batch[i] = resize_3d_image(moving_image, moving_image_shape)
            fixed_images_batch[i] = resize_3d_image(fixed_image, fixed_image_shape)
            
            if with_label_inputs:
                moving_labels_batch[i] = resize_3d_image(moving_label[:, :, :, label_to_select], moving_image_shape)
                fixed_labels_batch[i] = resize_3d_image(fixed_label[:, :, :, label_to_select], fixed_image_shape)
        
        zero_phis = np.zeros([batch_size, *moving_image_shape[:-1], 3])
        
        if with_label_inputs:
            inputs = [moving_images_batch, fixed_images_batch, moving_labels_batch, fixed_labels_batch]
            outputs = [fixed_images_batch, fixed_labels_batch, zero_phis]
        else:
            inputs = [moving_images_batch, fixed_images_batch]
            outputs = [fixed_images_batch, zero_phis]
        
        yield (inputs, outputs)
