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
         .astype({'Diabetes_binary':'uint8',
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

df = pd.read_csv("/home/g0nov4/Universidad/INF-354/PROYECTO_FINAL/diabetes_binary_health_indicators_BRFSS2015.csv")

df = data_clean(df)

print('### ANALISIS DE DATOS')
## Verificacion de valores faltantes
print(df.isnull().sum())


for column in df.columns:
    unique_values = df[column].unique()
    print("Unique values for column", column, ":")
    print(unique_values)
    print()



### PREPROCESAMIENTO
## Manejo de datos faltantes
# Imputación de valores faltantes utilizando la media
df = df.fillna(df.mean())

## Categorizacion de variables categoricas
# Codificación one-hot de variables categóricas
encoded_data = pd.get_dummies(df)
print(encoded_data)

## Normalizacion/escalado de variables
from sklearn.preprocessing import MinMaxScaler

# Crear un objeto Scaler
scaler = MinMaxScaler()


