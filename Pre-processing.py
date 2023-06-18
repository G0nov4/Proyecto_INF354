# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


def data_clean(dataframe):
    
    bins = [0, 18.5, 24.9, 29.9, float('inf')]
    labels = [1,2,3,4]
    
    df = (dataframe
         .drop_duplicates()
         .assign(BMI_bins=dataframe.BMI.pipe(pd.cut, bins=bins, labels=labels))
         .reset_index(drop=True)
         .astype({'Diabetes_012':'uint8',
     'HighBP':'uint8',
     'HighChol':'uint8',
     'CholCheck':'uint8',
     'BMI':'uint8',
     'Smoker':'uint8',
     'Stroke':'uint8',
     'HeartDiseaseorAttack':'uint8',
     'PhysActivity':'uint8',
     'Fruits':'uint8',
     'Veggies':'uint8',
     'HvyAlcoholConsump':'uint8',
     'AnyHealthcare':'uint8',
     'NoDocbcCost':'uint8',
     'GenHlth':'uint8',
     'MentHlth':'uint8',
     'PhysHlth':'uint8',
     'DiffWalk':'uint8',
     'Sex':'uint8',
     'Age':'uint8',
     'Education':'uint8',
     'Income':'uint8',
     'BMI_bins':'uint8'}) 
          
          
         )
    return df

df = pd.read_csv("/home/g0nov4/Universidad/INF-354/PROYECTO_FINAL/diabetes_012_health_indicators_BRFSS2015.csv")

df = data_clean(df)

# Principal information about Diabetes Dataset
print(df.head(), df.shape)

# Chech Info about dataset
#df.info()

# Check for values missing
print("REVISAMOS SI HAY VALORES FALTANTES")
print(df.isnull().sum())

print(df['Diabetes_012'].value_counts())

# VARIABLES HO HAS: 0=NO DIABETES, 1=PRE-DIABETES, 2=DIABETES CONFIRMED
df_no = df[df['Diabetes_012'] == 0]
df_pre = df[df['Diabetes_012'] == 1]
df_yes = df[df['Diabetes_012'] == 2]


# PRINT RANGE OF PEOPLE WHO HAS DIABETS
print("IMPRIMIMOS LOS RANGOS DE EDAD PARA LA DIABETES")
ax = sns.countplot(data=df_yes, x='Age')
ax.set(title= 'Distribucion de diabetes por años')
ax.set_xticklabels(['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '>80'], rotation = 45)
plt.show()




# Lets discover gender distribution 
""" fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=(20,10))
ax1 = sns.countplot(data=df_no, x='Sex', ax=ax1, palette='husl')
ax1.set(title='Gender distribution for no-diabetes')
ax1.set_xticklabels(['Female', 'Male'])

ax2 = sns.countplot(data=df_pre, x='Sex', ax=ax2, palette='husl')
ax2.set(title='Gender distribution for pre-diabetes')
ax2.set_xticklabels(['Female', 'Male'])

ax3 = sns.countplot(data=df_yes, x='Sex', ax=ax3, palette='husl')
ax3.set(title='Gender distribution for diabetics')
ax3.set_xticklabels(['Female', 'Male'])
plt.show() """


# Agrupacion
# grouped variables
target = 'Diabetes_012'
bool_vars = (df.nunique()[df.nunique() == 2]
                .index
                .drop(labels='Diabetes_012'))
num_vars = [var for var in df.columns if var not in bool_vars and var != 'Diabetes_012']

# OBEJTIVO
print(df['Diabetes_012'].value_counts(ascending=True))
print(df['Diabetes_012'].value_counts(1,ascending=True).apply(lambda x: format(x, '%')))
print()
df['Diabetes_012'].value_counts(1).plot(kind='barh',figsize=(10, 2)).spines[['top', 'right']].set_visible(False);
plt.title('Diabetes Objetivo 0-1-2 (%)', fontsize=18)
plt.yticks(ticks=[0,1,2], labels=['No-Diabetico','Pre-diabetes','Diabetico']);
plt.show()

# CARACTERSITICAS
colors = ['#be4d25','#2596be']
def analyse_cat(var):
    (df.groupby('Diabetes_012')[var]
     .value_counts(1)
     .unstack()
     .iloc[:,::-1]
     .plot(kind='barh',stacked=True,figsize=(10, 2), color=colors, alpha=1)
     .spines[['top', 'right']].set_visible(False))
    plt.legend(['Yes', "No"],bbox_to_anchor=(1, 1, 0, 0),shadow=False, frameon=False)
    plt.yticks(ticks=[0,1], labels=['Non-Diabetic', 'Diabetic'])
    plt.tight_layout()
    plt.title(var, fontsize=18)
    plt.show()

for var in bool_vars:
    analyse_cat(var)