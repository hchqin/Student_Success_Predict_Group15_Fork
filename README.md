# 🎓 Student Success Predictor 🎓

**Authors** (in alphabetical order): Katherine Chen, Hancheng Qin, Yili Tang, Bill Wan

This project aims to build a machine learning model to predict student's academic success.

## 📚 About

The Student Success Predictor project addresses the critical issue of academic dropout and failure in higher education. The dataset was meticulously created with the primary goal of leveraging machine learning techniques to identify students at risk early in their academic journey. The ultimate aim is to implement targeted strategies and support systems that contribute to the reduction of academic dropout. Throughout the project, we built machine learning models, like Support Vector Machines (SVM), Random Forest, and Logistic Regression (with L1 and L2 regularization), to predict if a student might drop out.

Due to a large number of features and their inter-correlations, our initial models exhibited signs of overfitting. We therefore incorporated feature selection techniques such as Principal Component Analysis (PCA) and feature importance analysis, coupled with fine-tuning the models' parameters. The refined models demonstrated enhanced performance, evident in a minimized gap between training and validation accuracy. Among the three models, SVM marginally outperformed the others, achieving an accuracy of 80% and an AUC score of 0.89. However, there remains potential for further improvement in model performance through additional feature engineering and more comprehensive parameter tuning.

Our dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

## Analysis

In the src directory, you will find four Jupyter notebooks: data_analysis_final_report.ipynb, data_analysis_model.ipynb, data_analysis_EDA.ipynb, and data_analysis_parameter_optimization.ipynb. For a comprehensive view of the analysis, execute data_analysis_final_report.ipynb, which integrates all individual parts. If you're interested in the specifics of each analytical segment, the other notebooks can be run separately to explore each in more detail.

## 📄 Report

The final report will be available upon completion of the project. [(link)] (https://github.com/UBC-MDS/DSCI522_Group15/blob/main/src/data_analysis_final_report.html)

## 💻 Usage

For the first time running the project, run the following from the root of this repository:

```bash
conda env create --file environment.yaml --name student_success_predictor
```

To run the analysis, activate the conda environment and start Jupyter Lab:

```         
conda activate student_success_predictor
jupyter lab 
```

Open the jupyter lab and run the [analysis file](https://github.com/UBC-MDS/DSCI522_Group15/blob/main/src/data_analysis_final_report.ipynb).

## 📦 Dependencies

-   conda (version 23.9.0 or higher)
-   nb_conda_kernels (version 2.3.1 or higher)
-   Python and packages listed in environment.yml

## 📜 License
The Student Success Predictor materials here are licensed under under MIT License. If re-using/re-mixing please provide attribution and link to this webpage.

## 📚 References
Please refer to the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) for the dataset used in this project.

- Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).
- Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215–232.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. DOI: 10.1023/A:1010933404324
- Pedregosa, F. et al., 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), pp.2825–2830.


