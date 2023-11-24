import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import altair as alt



from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA


student_df = pd.read_csv('../data/student.csv')


student_df.head(10)

print("The dataset has {} rows and {} columns. The target variable is {}".format(student_df.shape[0], student_df.shape[1], student_df.columns[-1]))

student_df.columns


train_df_org, test_df_org = train_test_split(student_df, test_size=0.2, random_state = 123)


train_df = train_df_org.copy()
test_df = test_df_org.copy()


train_df.sort_index().head()

train_df.info()


# ## EDA - Univariate Analysis:

### Map the marital status from code to actual status
### Categorical variable
status_mapping = {
    1: 'single',
    2: 'married',
    3: 'widower',
    4: 'divorced',

    5: 'facto union',
    6: 'legally separated'
}


train_df['Marital status'] = train_df['Marital status'].map(status_mapping)
test_df['Marital status'] = test_df['Marital status'].map(status_mapping)


### Map the Course from code to actual Course type easier for interpretation
### Categorical variable

course_mapping = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (evening attendance)'
}

# Apply the mapping
train_df['Course'] = train_df['Course'].map(course_mapping)
test_df['Course'] = test_df['Course'].map(course_mapping)

### Map the Course day/night attendance type easier for interpretation
### binary variable

course_mapping = {
    0: 'evening',
    1: 'daytime',
}

# Apply the mapping
train_df['Daytime/evening attendance\t'] = train_df['Daytime/evening attendance\t'].map(course_mapping)
test_df['Daytime/evening attendance\t'] = test_df['Daytime/evening attendance\t'].map(course_mapping)
train_df.rename(columns={'Daytime/evening attendance\t': 'Daytime evening attendance'}, inplace=True)
test_df.rename(columns={'Daytime/evening attendance\t': 'Daytime evening attendance'}, inplace=True)


### Map the nationality from code to actual nationality easier for interpretation
### Categorical variable

nation_mapping = {
    1: 'Portuguese',
    2: 'German',
    6: 'Spanish',
    11: 'Italian',
    13: 'Dutch',
    14: 'English',
    17: 'Lithuanian',
    21: 'Angolan',
    22: 'Cape Verdean',
    24: 'Guinean',
    25: 'Mozambican',
    26: 'Santomean',
    32: 'Turkish',
    41: 'Brazilian',
    62: 'Romanian',
    100: 'Moldova (Republic of)',
    101: 'Mexican',
    103: 'Ukrainian',
    105: 'Russian',
    108: 'Cuban',
    109: 'Colombian'
}

# Apply the mapping
train_df['Nacionality'] = train_df['Nacionality'].map(nation_mapping)
test_df['Nacionality'] = test_df['Nacionality'].map(nation_mapping)

### We divide our columns into numeric and categorical features this is our original version
numeric_features = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment', 
       'Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)', 
       'Unemployment rate',
       'Inflation rate', 
       'GDP']
categorical_features = ['Marital status', 
                        'Application mode', 
                        'Course', 
                        'Nacionality', 
                        "Mother's occupation", 
                        "Father's occupation"]
ordinal_features = ['Application order', 
                    'Previous qualification', 
                    "Mother's qualification", 
                    "Father's qualification"]
binary_features = ['Daytime evening attendance', 
                   'Displaced', 
                   'Educational special needs', 
                   'Debtor', 
                   'Tuition fees up to date', 
                   'Gender', 
                   'Scholarship holder', 
                   'International']
drop_features = []
target = "Target"

train_df["Target"].value_counts()

train_df.loc[:, numeric_features].describe()


train_df['Target'].value_counts()

n_features = len(categorical_features[:3] + binary_features)
n_rows = (n_features + 1) // 2

if n_features > 0:
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows))

    for i, feat in enumerate(categorical_features[:3] + binary_features):
        row = i // 2
        col = i % 2
        sns.countplot(data=train_df, x=feat, hue="Target", palette="Set2", alpha=0.6, ax=axes[row, col])
        axes[row, col].set_title("Distribution of " + feat + " by Target", fontweight='bold')
        axes[row, col].set_ylabel("Count", fontweight='bold')
        
        # Rotate x-axis labels at a 45-degree angle without triggering the warning
        
        axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), rotation=45, ha='right')

    # If there is an odd number of features, remove the last subplot
    if n_features % 2 == 1:
        fig.delaxes(axes[-1, -1])

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
else:
    print("No features to plot.")


n_features = len(numeric_features)
n_rows = (n_features + 1) // 2  

fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows))

for i, feat in enumerate(numeric_features):
    row = i // 2
    col = i % 2
    sns.histplot(data=train_df, x=feat, hue="Target", kde=True, palette="Set2", element="step", ax=axes[row, col])
    axes[row, col].set_title("Distribution of " + feat + " by Target", fontweight='bold')
    axes[row, col].set_xlabel(feat, fontweight='bold')
    axes[row, col].set_ylabel("Density", fontweight='bold')

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# ### Comments Categorical Variable
# 1. Nationality:
# The majority are Portuguese, with a small representation from other nationalities. 
# 
# 2. Parents' Occupation:
# Both mother's and father's occupations are coded numerically. The most common occupation code for mothers and fathers is Unskilled Workers. Be careful about the matrix sparsity issue.
# 
# 3. Debtor, Tuition Fees Up to Date, Scholarship Holder:
# There's a notable number of students who are debtors (397) or whose tuition fees are not up to date (419), while 871 are scholarship holders. These figures highlight the financial aspects and challenges faced by the student population.

# ### Comments Numeric Variable:
# 1. Previous Qualification (Grade) and Admission Grade:
# - Both these variables have similar ranges (min 95 to max 190), indicating a possible correlation between previous academic performance and admission grades.
# - The mean and median values are close, suggesting a relatively symmetric distribution for these variables.
# 
# 2. Age at Enrollment:
# - The age range is quite broad (17 to 70 years), indicating a diverse set of students in terms of age. Transformation technique like Standardization is required
# 
# 3. Curricular Units Credited (1st and 2nd Semesters):
# - The mean values for credited curricular units in both semesters are low (around 0.71 for the 1st semester and 0.54 for the 2nd), the 75% percentile is 0, suggesting that most students do not have many, if any, units credited. This could be because they are first year students.

# ## EDA - Bivariate Analysis:


numeric_subset = train_df.loc[:, numeric_features]
numeric_subset.corr('spearman').style.background_gradient(cmap='viridis')


alt.Chart(numeric_subset).mark_point(opacity=0.3, size=5).encode(
     alt.X(alt.repeat('row'), type='quantitative', scale=alt.Scale(zero=False)),
     alt.Y(alt.repeat('column'), type='quantitative', scale=alt.Scale(zero=False))
).properties(
    width=150,
    height=150
).repeat(
    column=list(numeric_subset.columns),
    row=list(numeric_subset.columns)
)

numeric_subset.head()

def calculate_correlation_df(dataframe, threshold=0.5):
    """
    Calculate correlations between numeric variables in a dataframe using Spearman method.

    Parameters:
    - dataframe: pandas DataFrame
    - threshold: float, correlation threshold for filtering (default is 0.5)

    Returns:
    - pandas DataFrame containing Variable 1, Variable 2, and Correlation columns for significant correlations.
    """
    numeric_subset = dataframe.select_dtypes(include=[np.number])
    corr_matrix = numeric_subset.corr(method='spearman')

    result_corr = dict()
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if np.abs(corr_matrix.loc[i, j]) >= threshold and np.abs(corr_matrix.loc[i, j]) <= 1:
                temp_value = corr_matrix.loc[i, j]
                if (j, i) not in result_corr:
                    result_corr[(i, j)] = temp_value

    data_list = [(key[0], key[1], value) for key, value in result_corr.items()]
    corr_df = pd.DataFrame(data_list, columns=['Variable 1', 'Variable 2', 'Correlation'])

    return corr_df


result_dataframe = calculate_correlation_df(train_df)
print(result_dataframe)
        
