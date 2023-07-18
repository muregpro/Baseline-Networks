import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage import generate_binary_structure, distance_transform_edt, binary_erosion

from data_generator import resize_3d_image
from tqdm import tqdm

# import SimpleITK as sitk

# adapted from voxelmorph
def DSC(y_true, y_pred):
        
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims + 1))

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

    div_no_nan = tf.math.divide_no_nan if hasattr(
        tf.math, 'divide_no_nan') else tf.div_no_nan  
    dice = tf.reduce_mean(div_no_nan(top, bottom))
    return dice

def RDSC(y_true, y_pred):
        
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims + 1))

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

    div_no_nan = tf.math.divide_no_nan if hasattr(
        tf.math, 'divide_no_nan') else tf.div_no_nan  
    dice_scores_all = div_no_nan(top, bottom)
    sorted_indices = tf.argsort(dice_scores_all, axis=0, direction='DESCENDING')
    num_samples_dsc = tf.convert_to_tensor(tf.math.round(len(sorted_indices) * 0.68))
    dsc_scores_for_robustness = tf.gather(dice_scores_all, sorted_indices[:tf.cast(num_samples_dsc, dtype=tf.int32)])
    rdsc = tf.reduce_mean(tf.squeeze(dsc_scores_for_robustness))
    return rdsc

def compute_centroids(y_true, y_pred):    
    # find centroid locations in y_true
    y_true_centroids = []
    for i in range(len(y_true)):
        y_true_ones_indices = tf.where(y_true[i] > 0.5)
        image_centroid = tf.math.reduce_mean(y_true_ones_indices[:, :-1], axis=0)
        y_true_centroids.append(image_centroid)
    y_true_centroids = tf.stack(y_true_centroids)
    
    #find centroid locations in y_pred
    y_pred_centroids = []
    for i in range(len(y_pred)):
        y_pred_ones_indices = tf.where(y_pred[i] > 0.5)
        image_centroid = tf.math.reduce_mean(y_pred_ones_indices[:, :-1], axis=0)
        y_pred_centroids.append(image_centroid)
    y_pred_centroids = tf.stack(y_pred_centroids)
    return y_true_centroids, y_pred_centroids

def centroid_maes(y_true, y_pred):    
    y_true_centroids, y_pred_centroids = compute_centroids(y_true, y_pred)
    maes = tf.keras.losses.MAE(y_true_centroids, y_pred_centroids)
    return maes

def RMS(tensor):
    return tf.math.sqrt(tf.cast(tf.math.reduce_mean(tf.math.square(tensor)), dtype=tf.float32))

def TRE(all_label_maes):
    TREs = []
    for i in range(len(all_label_maes)):
        case = all_label_maes[i]
        case_rms = RMS(case)
        TREs.append(case_rms)
    mTRE = tf.math.reduce_mean(tf.stack(TREs))
    return mTRE

def RTRE(all_label_maes):
    TREs = []
    for i in range(len(all_label_maes)):
        case = all_label_maes[i]
        case_rms = RMS(case)
        TREs.append(case_rms)
    TREs =tf.stack(TREs)
    
    sorted_indices = tf.argsort(TREs, axis=0, direction='ASCENDING')
    num_samples_tre = tf.convert_to_tensor(tf.math.round(len(sorted_indices) * 0.68))
    # tres_for_robustness = TREs[0:tf.cast(num_samples_tre, dtype=tf.int32)]
    tres_for_robustness = tf.gather(TREs, sorted_indices[:tf.cast(num_samples_tre, dtype=tf.int32)])
    rtre = tf.reduce_mean(tf.squeeze(tres_for_robustness))
    return rtre

def RTs(all_label_maes):
    TREs = []
    for i in range(len(all_label_maes)):
        case = all_label_maes[i]
        
        sorted_indices = tf.argsort(case, axis=0, direction='ASCENDING')
        num_samples_case_rts = tf.convert_to_tensor(tf.math.round(len(sorted_indices) * 0.68))
        # case_for_rts = case[0:tf.cast(num_samples_case_rts, dtype=tf.int32)]
        case_for_rts = tf.gather(case, sorted_indices[:tf.cast(num_samples_case_rts, dtype=tf.int32)])
        
        case_rms = RMS(case_for_rts)
        TREs.append(case_rms)
    mTRE = tf.math.reduce_mean(tf.stack(TREs))
    return mTRE

# adapted from loli/medpy
def surface_distances(result, reference, voxelspacing=None, connectivity=1):
    
    result = resize_3d_image(result, np.shape(reference))
    
    result = np.atleast_1d(np.array(result).astype(bool))
    reference = np.atleast_1d(np.array(reference).astype(bool))
    
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

# adapted from loli/medpy
def HD95(y_true, y_pred):
    
    all_hd95s = []
    print('computing hd')
    for i in tqdm(range(len(y_true))):
        
        if np.shape(y_true[i]) != np.shape(y_pred[i]):
            y_pred_to_use = resize_3d_image(y_pred[i:i+1], np.shape(y_true[i:i+1]))
        else:
            y_pred_to_use = y_pred[i:i+1]
        
        # print(np.shape(y_pred_to_use), np.shape(y_true[i:i+1]))
        
        voxelspacing = None
        connectivity = 1
        hd1 = surface_distances(y_pred_to_use, y_true[i:i+1], voxelspacing, connectivity)
        hd2 = surface_distances(y_true[i:i+1], y_pred_to_use, voxelspacing, connectivity)
        try:
            hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
        except IndexError:
            hd95 = 0.0
        all_hd95s.append(hd95)
    return np.mean(all_hd95s)


# def StDJD(ddf):
#     ddf_sitk = sitk.GetImageFromArray(ddf)
#     ddf_sitk.SetOrigin((0, 0, 0))
#     jacobian_determinant = sitk.DisplacementFieldJacobianDeterminant(ddf_sitk)
#     std_jacobian_determinant = np.std(sitk.GetArrayFromImage(jacobian_determinant))
#     return std_jacobian_determinant