from enum import Enum
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
import pandas as pd

class Encoders(Enum):
    LABEL = 'Label Encoder',
    ONE_HOT = 'One-Hot Encoder'

    def __init__(self, description):
       self.description = description

    @classmethod
    def values(self):
        return [x.description for x in Encoders]

    @classmethod
    def get(self, description):
        result =  [x for x in Encoders if x.description == description]
        return None if len(result) == 0 else result[0]

    def create_encoder(self) -> TransformerMixin:
        if self == Encoders.ONE_HOT:
            return OneHotEncoder()
        elif self == Encoders.LABEL:
            return LabelEncoder()
        else:
            raise TypeError('Tipo de encoder não suportado.')
    
    def fit_transform(self, df:pd.DataFrame, cols:list[str]) -> pd.DataFrame:
        if self == Encoders.ONE_HOT:
            return self.__one_hot_transform(df, cols)
        elif self == Encoders.LABEL:
            return self.__label_transform(df, cols)
        else:
            raise TypeError('Tipo de encoder não suportado.')
        
    def __one_hot_transform(self, df, cols):
        encoders = [(self.create_encoder(), [col]) for col in cols]
        transformer = make_column_transformer(
            *encoders,
            remainder='passthrough'
        )
        transformer = transformer.fit(df)
        result = pd.DataFrame(transformer.transform(df),
                           columns=transformer.get_feature_names_out())
        return result

    def __label_transform(self, df:pd.DataFrame, cols:list[str]):
        for c in cols:
            df[c] = LabelEncoder().fit_transform(df[[c]])
        return df
    
    def __str__(self) -> str:
        return self.description
    
class Scalers(Enum):
    STANDARD = 'Padrão (Z-Score)', 
    MINMAX = 'Normalização Min-Max'

    def __init__(self, description):
       self.description = description

    @classmethod
    def values(self):
        return [x.description for x in Scalers]

    @classmethod
    def get(self, description:str):
        result =  [x for x in Scalers if x.description == description]
        return None if len(result) == 0 else result[0]

    def create_scaler(self) -> TransformerMixin:
        if self == Scalers.STANDARD:
            return StandardScaler()
        elif self == Scalers.MINMAX:
            return MinMaxScaler()
        else:
            raise TypeError('Tipo de scaler não suportado.')
    
    def fit_transform(self, df:pd.DataFrame, col:str) -> pd.DataFrame:
        #observe que para os scalers precisa do reshape, mas para os encoders não.
        vals = df[col].to_numpy().reshape(-1,1)
        scaler = self.create_scaler()
        vals_enc = scaler.fit_transform(vals)
        df_scaled = pd.DataFrame(vals_enc, columns=[col])
        df[col] = df_scaled[col]
        return df

    def __str__(self) -> str:
        return self.description

