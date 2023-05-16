FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN mkdir -p ./mureg

ADD ./ ./mureg

WORKDIR mureg

RUN pip install joblib scikit-image==0.20.0 nibabel==5.1.0 matplotlib==3.7.1 voxelmorph==0.2
