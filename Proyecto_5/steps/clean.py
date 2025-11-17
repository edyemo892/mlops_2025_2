# 💡 Comentar varias líneas en VS Code
# ✅ Para comentar líneas completas (comentario de línea):
# - Windows/Linux:
# Selecciona las líneas → Presiona Ctrl + K seguido de Ctrl + C
# (esto comenta todas las líneas seleccionadas)

# Para descomentar: Ctrl + K seguido de Ctrl + U
# ✅ Para comentar con bloques (comentario de bloque):




#import packages for data manipulation
import pandas as pd
import numpy as np

#import packages for machine learning
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

#import packages for data management
import joblib
class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        
        
    def clean_data(self, data):
        # Normalizar nombres de columnas a str por seguridad
        data = data.copy()

        #use ordinal encoder to encode experience level
        encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
        data['experience_level_encoded'] = encoder.fit_transform(data[['experience_level']])

        #use ordinal encoder to encode company size
        encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
        data['company_size_encoded'] = encoder.fit_transform(data[['company_size']])

        #encode employmeny type and job title using dummy columns
        #La llevo a train.py para respetar el flujo de trabajo
        #data = pd.get_dummies(data, columns = ['employment_type', 'job_title'], drop_first = True, dtype = int)

        #drop original columns
        data = data.drop(data.columns[0], axis=1)
        data = data.drop(columns = ['experience_level', 'company_size','work_year','salary','salary_currency','employee_residence','remote_ratio','company_location'])
        
        # data.drop(['id','SalesChannelID','VehicleAge','DaysSinceCreated'], axis=1, inplace=True)
        
        # data['AnnualPremium'] = data['AnnualPremium'].str.replace('£', '').str.replace(',', '').astype(float)
            
        # for col in ['Gender', 'RegionID']:
        #      data[col] = self.imputer.fit_transform(data[[col]]).flatten()
             
        # data['Age'] = data['Age'].fillna(data['Age'].median())
        # data['HasDrivingLicense']= data['HasDrivingLicense'].fillna(1)
        # data['Switch'] = data['Switch'].fillna(-1)
        # data['PastAccident'] = data['PastAccident'].fillna("Unknown", inplace=False)
        
        # Q1 = data['AnnualPremium'].quantile(0.25)
        # Q3 = data['AnnualPremium'].quantile(0.75)
        # IQR = Q3 - Q1
        # upper_bound = Q3 + 1.5 * IQR
        # data = data[data['AnnualPremium'] <= upper_bound]
        
        return data