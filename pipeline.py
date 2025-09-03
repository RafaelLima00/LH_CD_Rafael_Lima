
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

# Carregar os dados brutos
url = "https://raw.githubusercontent.com/RafaelLima00/LH_CD_Rafael_Lima/main/data/desafio_indicium_imdb.csv"
df = pd.read_csv(url)

# Limpeza inicial dos dados
df = df.drop(columns=['Unnamed: 0'])
df['Runtime'] = df['Runtime'].str.replace(' min', '', regex=False)
df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
df['Gross'] = df['Gross'].str.replace(',', '', regex=False)
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')

# Classe para criar novas features
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        genres = X_copy['Genre'].str.split(', ', expand=True)
        for i in range(genres.shape[1]):
            X_copy[f'Genre {i+1}'] = genres[i]
        X_copy['Num_Genres'] = X_copy['Genre'].str.split(',').apply(len)
        X_copy['Score_Diff'] = X_copy['IMDB_Rating'] - (X_copy['Meta_score'] / 10.0)
        return X_copy

# Classe para combinar múltiplas colunas de texto em uma única Series
class ColumnCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns].agg(' '.join, axis=1)

# Pipelines individuais para cada tipo de dado
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
text_transformer = Pipeline(steps=[
    ('combiner', ColumnCombiner(columns=['Director', 'Star1', 'Star2', 'Star3', 'Star4'])),
    ('tfidf', TfidfVectorizer(max_features=80))
])

# Juntar todos os pipelines com o ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Runtime', 'Meta_score', 'No_of_Votes', 'Gross', 'Num_Genres', 'Score_Diff']),
        ('cat', categorical_transformer, ['Certificate', 'Genre 1', 'Genre 2', 'Genre 3']),
        ('text', text_transformer, ['Director', 'Star1', 'Star2', 'Star3', 'Star4'])
    ],
    remainder='drop'
)

# Separar os dados em treino e teste
X = df.drop('IMDB_Rating', axis=1)
y = df['IMDB_Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Construir o pipeline final
final_pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineer()),
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=32))
])

# Treinar o pipeline
final_pipeline.fit(X_train, y_train)

# Avaliar o modelo
y_pred = final_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("--- Resultados da Avaliação do Pipeline ---")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Salvar o pipeline
with open('final_pipeline.pkl', 'wb') as f:
    pickle.dump(final_pipeline, f)

# Carregar o pipeline
with open('final_pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Fazer uma previsão com novos dados
exemplo = {
    'Series_Title': 'The Shawshank Redemption',
    'Released_Year': '1994',
    'Certificate': 'A',
    'Runtime': 142.0,
    'Genre': 'Drama',
    'Overview': 'Two imprisoned men bond over a number of years...',
    'Meta_score': 80.0,
    'Director': 'Frank Darabont',
    'Star1': 'Tim Robbins', 'Star2': 'Morgan Freeman', 'Star3': 'Bob Gunton', 'Star4': 'William Sadler',
    'No_of_Votes': 2343110,
    'Gross': 28341469.0,
    'IMDB_Rating': 9.3
}
exemplo_df = pd.DataFrame([exemplo])
previsao_nota = loaded_pipeline.predict(exemplo_df)
print("\n--- Previsão com o Pipeline ---")
print(f"Nota IMDB Prevista: {previsao_nota[0]:.2f}")
```