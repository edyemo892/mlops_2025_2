# 💡 Comentar varias líneas en VS Code
# ✅ Para comentar líneas completas (comentario de línea):
# - Windows/Linux:
# Selecciona las líneas → Presiona Ctrl + K seguido de Ctrl + C
# (esto comenta todas las líneas seleccionadas)

# Para descomentar: Ctrl + K seguido de Ctrl + U
# ✅ Para comentar con bloques (comentario de bloque):


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        self.ohe = OneHotEncoder(
            categories=[
                [
                    "Data Engineer",
                    "Data Manager",
                    "Data Scientist",
                    "Machine Learning Engineer"
                ]
            ],
            handle_unknown="ignore",
            sparse_output=False
        )

    def clean_data(self, data):
        data = data.copy()

        # 1. Encode ordinal experience
        encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
        data['experience_level_encoded'] = encoder.fit_transform(data[['experience_level']])

        # 2. Encode ordinal company size
        encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
        data['company_size_encoded'] = encoder.fit_transform(data[['company_size']])

        # 3. One-hot encoding
       
        # 1. Separar las columnas a codificar
        ohe_input = data[["job_title"]]

        # 2. Ajustar y transformar
        ohe_output = self.ohe.fit_transform(ohe_input)

        # 3. Crear nombres para las columnas especificadas
        ohe_cols = (
           [
                "job_title_Data_Engineer",
                "job_title_Data_Manager",
                "job_title_Data_Scientist",
                "job_title_Machine_Learning_Engineer"
            ]
        )

        df_ohe = pd.DataFrame(ohe_output, columns=ohe_cols, index=data.index)

        # 4. Agregar al dataset original
        data = pd.concat([data, df_ohe], axis=1)
        data = data.drop(columns=[ "job_title"])

        data["employment_type_PT"] = (data["employment_type"] == "PT").astype(int)

        # 4. Drop columnas no necesarias
        data = data.drop(data.columns[0], axis=1)  # Unnamed: 0
        data = data.drop(columns=[
            'employment_type',
            'experience_level',
            'company_size',
            'work_year',
            'salary',
            'salary_currency',
            'employee_residence',
            'remote_ratio',
            'company_location'
        ])
        desired_order = [
            "salary_in_usd",
            "experience_level_encoded",
            "company_size_encoded",
            "employment_type_PT",
            "job_title_Data_Engineer",
            "job_title_Data_Manager",
            "job_title_Data_Scientist",
            "job_title_Machine_Learning_Engineer"
        ]

        data = data[desired_order]
        return data

    def split_data(self, data_clean, split_ratio=0.8):
        split = int(len(data_clean) * split_ratio)
        train_data = data_clean.iloc[:split].reset_index(drop=True)
        test_data  = data_clean.iloc[split:].reset_index(drop=True)
        return train_data, test_data