#author bill wan
FROM quay.io/jupyter/minimal-notebook:2023-11-19

RUN conda install -y pandas=2.1.2 \
    numpy=1.26\
    scikit-learn=1.3.2 \
    altair=5.1.2 \
    imbalanced-learn \
    matplotlib=3.8.0\
    scipy=1.11.3\
    seaborn=0.13.0\
    jupyter=1.0.0\
    nbconvert=7.11.0\
    pandoc\
    python=3.11.* \
    pytest=7.4.3\
    click=8.1.7\
    vl-convert-python=1.1.0 \
    jupyter-book=0.15.1

RUN pip install myst-nb==0.17.2