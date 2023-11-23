FROM quay.io/jupyter/minimal-notebook:2023-11-19

RUN conda install -y pandas=2.1.2 \
    numpy\
    scikit-learn=1.3.2 \
    altair=5.1.2 \
    imbalanced-learn \
    matplotlib=3.8.0\
    scipy=1.11.3\
    seaborn\
    jupyter=1.0.0\
    nbconvert=7.11.0\
    pandoc\
    python \
    pytest=7.4.3
    