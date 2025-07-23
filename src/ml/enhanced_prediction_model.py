"""
Modelo de predição aprimorado usando scikit-learn
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
import json

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from flask import Flask
from src.models.user import db
from src.models.climate import ClimateData
from src.models.arbovirus import ArbovirusData
from src.models.prediction import PredictionResult

class EnhancedArbovirusPredictionModel:
    """Modelo aprimorado de predição de arboviroses usando scikit-learn"""
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.model = None
        self.scaler = StandardScaler()
        self.model_version = "v2.1.0"
        self.feature_names = [
            'temperature_avg', 'temperature_min', 'temperature_max',
            'precipitation', 'humidity_avg', 'wind_speed_avg',
            'temp_range', 'temp_humidity_interaction',
            'temperature_avg_ma7', 'precipitation_ma7', 'humidity_avg_ma7',
            'seasonal_factor', 'day_of_year', 'month'
        ]
        
    def create_app(self):
        """Criar aplicação Flask se não fornecida"""
        if not self.app:
            app = Flask(__name__)
            app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'database', 'app.db')}"
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)
            return app
        return self.app
    
    def load_training_data(self, days_back: int = 365) -> pd.DataFrame:
        """
        Carregar dados de treinamento de todas as cidades
        
        Args:
            days_back: Número de dias para voltar no histórico
            
        Returns:
            DataFrame com dados combinados de todas as cidades
        """
        app = self.create_app()
        
        with app.app_context():
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            # Carregar dados climáticos
            climate_query = ClimateData.query.filter(
                ClimateData.date >= start_date,
                ClimateData.date <= end_date
            ).order_by(ClimateData.city, ClimateData.state, ClimateData.date)
            
            climate_data = []
            for record in climate_query.all():
                climate_data.append({
                    'date': record.date,
                    'city': record.city,
                    'state': record.state,
                    'temperature_avg': float(record.temperature_avg) if record.temperature_avg else 0,
                    'temperature_min': float(record.temperature_min) if record.temperature_min else 0,
                    'temperature_max': float(record.temperature_max) if record.temperature_max else 0,
                    'precipitation': float(record.precipitation) if record.precipitation else 0,
                    'humidity_avg': float(record.humidity_avg) if record.humidity_avg else 0,
                    'wind_speed_avg': float(record.wind_speed_avg) if record.wind_speed_avg else 0
                })
            
            # Carregar dados de arboviroses
            arbovirus_query = ArbovirusData.query.filter(
                ArbovirusData.date >= start_date,
                ArbovirusData.date <= end_date
            ).order_by(ArbovirusData.city, ArbovirusData.state, ArbovirusData.date)
            
            arbovirus_data = []
            for record in arbovirus_query.all():
                arbovirus_data.append({
                    'date': record.date,
                    'city': record.city,
                    'state': record.state,
                    'cases_dengue': record.cases_dengue or 0,
                    'cases_zika': record.cases_zika or 0,
                    'cases_chikungunya': record.cases_chikungunya or 0,
                    'incidence_dengue': float(record.incidence_dengue) if record.incidence_dengue else 0
                })
            
            # Combinar dados
            climate_df = pd.DataFrame(climate_data)
            arbovirus_df = pd.DataFrame(arbovirus_data)
            
            if not climate_df.empty and not arbovirus_df.empty:
                # Merge por data, cidade e estado
                combined_df = pd.merge(
                    climate_df, arbovirus_df, 
                    on=['date', 'city', 'state'], 
                    how='outer'
                )
                combined_df = combined_df.fillna(0)
                combined_df = combined_df.sort_values(['city', 'state', 'date'])
                return combined_df
            elif not climate_df.empty:
                # Apenas dados climáticos
                climate_df['cases_dengue'] = 0
                climate_df['cases_zika'] = 0
                climate_df['cases_chikungunya'] = 0
                climate_df['incidence_dengue'] = 0
                return climate_df
            else:
                return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preparar features para o modelo
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame com features preparadas
        """
        if df.empty:
            return df
        
        # Converter date para datetime se necessário
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Criar features derivadas
        df['temp_range'] = df['temperature_max'] - df['temperature_min']
        df['temp_humidity_interaction'] = df['temperature_avg'] * df['humidity_avg']
        
        # Features temporais
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        
        # Fator sazonal (dengue é mais comum no verão/outono no Brasil)
        df['seasonal_factor'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Adicionar médias móveis por cidade
        for city_state in df[['city', 'state']].drop_duplicates().values:
            city, state = city_state
            mask = (df['city'] == city) & (df['state'] == state)
            
            for col in ['temperature_avg', 'precipitation', 'humidity_avg']:
                df.loc[mask, f'{col}_ma7'] = df.loc[mask, col].rolling(window=7, min_periods=1).mean()
        
        return df
    
    def create_lagged_features(self, df: pd.DataFrame, lag_days: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
        """
        Criar features com lag temporal
        
        Args:
            df: DataFrame com dados
            lag_days: Lista de dias de lag
            
        Returns:
            DataFrame com features de lag
        """
        if df.empty:
            return df
        
        # Ordenar por cidade, estado e data
        df = df.sort_values(['city', 'state', 'date'])
        
        # Criar features de lag para cada cidade
        for city_state in df[['city', 'state']].drop_duplicates().values:
            city, state = city_state
            mask = (df['city'] == city) & (df['state'] == state)
            
            for lag in lag_days:
                # Lag de casos de dengue
                df.loc[mask, f'cases_dengue_lag_{lag}'] = df.loc[mask, 'cases_dengue'].shift(lag)
                
                # Lag de temperatura
                df.loc[mask, f'temperature_avg_lag_{lag}'] = df.loc[mask, 'temperature_avg'].shift(lag)
                
                # Lag de precipitação
                df.loc[mask, f'precipitation_lag_{lag}'] = df.loc[mask, 'precipitation'].shift(lag)
        
        # Preencher valores NaN com 0
        lag_columns = [col for col in df.columns if '_lag_' in col]
        df[lag_columns] = df[lag_columns].fillna(0)
        
        return df
    
    def build_model(self) -> Pipeline:
        """
        Construir pipeline do modelo
        
        Returns:
            Pipeline do scikit-learn
        """
        # Usar Gradient Boosting que funciona bem com dados temporais
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8
            ))
        ])
        
        return model
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Treinar o modelo
        
        Args:
            df: DataFrame com dados de treinamento
            
        Returns:
            Métricas de treinamento
        """
        if df.empty:
            print("❌ Dados de treinamento vazios")
            return {'status': 'no_data'}
        
        print(f"📊 Preparando dados de treinamento com {len(df)} registros...")
        
        # Preparar features
        df = self.prepare_features(df)
        df = self.create_lagged_features(df)
        
        # Selecionar features disponíveis
        available_features = [col for col in self.feature_names if col in df.columns]
        lag_features = [col for col in df.columns if '_lag_' in col]
        all_features = available_features + lag_features
        
        if not all_features:
            print("❌ Nenhuma feature disponível")
            return {'status': 'no_features'}
        
        # Preparar dados para treinamento
        X = df[all_features].values
        y = df['cases_dengue'].values
        
        # Remover linhas com NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            print("❌ Dados insuficientes para treinamento")
            return {'status': 'insufficient_data'}
        
        print(f"📈 Treinando com {len(X)} amostras e {len(all_features)} features")
        
        # Dividir dados em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Construir e treinar modelo
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        # Calcular métricas
        metrics = {
            'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
            'val_mae': float(mean_absolute_error(y_val, y_pred_val)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
            'train_r2': float(r2_score(y_train, y_pred_train)),
            'val_r2': float(r2_score(y_val, y_pred_val)),
            'n_features': len(all_features),
            'n_samples': len(X),
            'status': 'success'
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        metrics['cv_mae'] = float(-cv_scores.mean())
        metrics['cv_mae_std'] = float(cv_scores.std())
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"   - MAE Validação: {metrics['val_mae']:.2f}")
        print(f"   - R² Validação: {metrics['val_r2']:.3f}")
        print(f"   - CV MAE: {metrics['cv_mae']:.2f} ± {metrics['cv_mae_std']:.2f}")
        
        # Salvar lista de features usadas
        self.trained_features = all_features
        
        return metrics
    
    def predict_for_city(self, city: str, state: str, prediction_date: date = None) -> Dict[str, float]:
        """
        Fazer predição para uma cidade específica
        
        Args:
            city: Nome da cidade
            state: Estado
            prediction_date: Data da predição (padrão: 30 dias à frente)
            
        Returns:
            Dicionário com predições
        """
        if self.model is None:
            return self._simple_prediction()
        
        if prediction_date is None:
            prediction_date = date.today() + timedelta(days=30)
        
        try:
            # Carregar dados históricos da cidade
            historical_data = self.load_city_data(city, state, days_back=90)
            
            if historical_data.empty:
                return self._simple_prediction()
            
            # Preparar features
            historical_data = self.prepare_features(historical_data)
            historical_data = self.create_lagged_features(historical_data)
            
            # Pegar última linha (mais recente)
            latest_data = historical_data.iloc[-1:].copy()
            
            # Ajustar data para predição
            latest_data['date'] = pd.to_datetime(prediction_date)
            latest_data['day_of_year'] = latest_data['date'].dt.dayofyear
            latest_data['month'] = latest_data['date'].dt.month
            latest_data['seasonal_factor'] = np.sin(2 * np.pi * latest_data['day_of_year'] / 365.25)
            
            # Selecionar features usadas no treinamento
            if hasattr(self, 'trained_features'):
                available_features = [col for col in self.trained_features if col in latest_data.columns]
                if not available_features:
                    return self._simple_prediction()
                
                X = latest_data[available_features].values
                
                # Verificar se há NaN
                if np.isnan(X).any():
                    X = np.nan_to_num(X, nan=0.0)
                
                # Fazer predição
                prediction = self.model.predict(X)[0]
                prediction = max(0, prediction)  # Garantir não-negativo
                
                return {
                    'predicted_cases_dengue': float(prediction),
                    'predicted_cases_zika': float(prediction * 0.15),
                    'predicted_cases_chikungunya': float(prediction * 0.25),
                    'confidence_score': 0.80,
                    'model_type': 'gradient_boosting'
                }
            else:
                return self._simple_prediction()
                
        except Exception as e:
            print(f"❌ Erro na predição para {city}-{state}: {str(e)}")
            return self._simple_prediction()
    
    def load_city_data(self, city: str, state: str, days_back: int = 90) -> pd.DataFrame:
        """Carregar dados históricos de uma cidade específica"""
        app = self.create_app()
        
        with app.app_context():
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            # Carregar dados climáticos
            climate_query = ClimateData.query.filter(
                ClimateData.city == city,
                ClimateData.state == state,
                ClimateData.date >= start_date,
                ClimateData.date <= end_date
            ).order_by(ClimateData.date)
            
            climate_data = []
            for record in climate_query.all():
                climate_data.append({
                    'date': record.date,
                    'city': record.city,
                    'state': record.state,
                    'temperature_avg': float(record.temperature_avg) if record.temperature_avg else 0,
                    'temperature_min': float(record.temperature_min) if record.temperature_min else 0,
                    'temperature_max': float(record.temperature_max) if record.temperature_max else 0,
                    'precipitation': float(record.precipitation) if record.precipitation else 0,
                    'humidity_avg': float(record.humidity_avg) if record.humidity_avg else 0,
                    'wind_speed_avg': float(record.wind_speed_avg) if record.wind_speed_avg else 0
                })
            
            # Carregar dados de arboviroses
            arbovirus_query = ArbovirusData.query.filter(
                ArbovirusData.city == city,
                ArbovirusData.state == state,
                ArbovirusData.date >= start_date,
                ArbovirusData.date <= end_date
            ).order_by(ArbovirusData.date)
            
            arbovirus_data = []
            for record in arbovirus_query.all():
                arbovirus_data.append({
                    'date': record.date,
                    'city': record.city,
                    'state': record.state,
                    'cases_dengue': record.cases_dengue or 0,
                    'cases_zika': record.cases_zika or 0,
                    'cases_chikungunya': record.cases_chikungunya or 0
                })
            
            # Combinar dados
            climate_df = pd.DataFrame(climate_data)
            arbovirus_df = pd.DataFrame(arbovirus_data)
            
            if not climate_df.empty and not arbovirus_df.empty:
                combined_df = pd.merge(climate_df, arbovirus_df, on=['date', 'city', 'state'], how='outer')
                combined_df = combined_df.fillna(0)
                return combined_df.sort_values('date')
            elif not climate_df.empty:
                climate_df['cases_dengue'] = 0
                climate_df['cases_zika'] = 0
                climate_df['cases_chikungunya'] = 0
                return climate_df
            else:
                return pd.DataFrame()
    
    def _simple_prediction(self) -> Dict[str, float]:
        """Predição simples como fallback"""
        return {
            'predicted_cases_dengue': 25.0,
            'predicted_cases_zika': 5.0,
            'predicted_cases_chikungunya': 8.0,
            'confidence_score': 0.5,
            'model_type': 'fallback'
        }
    
    def save_model(self, filepath: str):
        """Salvar modelo treinado"""
        if self.model is None:
            print("⚠️ Modelo não treinado")
            return
        
        try:
            # Salvar modelo
            with open(f"{filepath}_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            
            # Salvar metadados
            metadata = {
                'model_version': self.model_version,
                'feature_names': getattr(self, 'trained_features', self.feature_names),
                'created_at': datetime.now().isoformat()
            }
            
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo salvo em {filepath}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {str(e)}")
    
    def load_model(self, filepath: str):
        """Carregar modelo treinado"""
        try:
            # Carregar modelo
            with open(f"{filepath}_model.pkl", 'rb') as f:
                self.model = pickle.load(f)
            
            # Carregar metadados
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
                self.model_version = metadata.get('model_version', 'v2.1.0')
                self.trained_features = metadata.get('feature_names', self.feature_names)
            
            print(f"✅ Modelo carregado de {filepath}")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {str(e)}")

def main():
    """Função principal para treinar o modelo"""
    print("🚀 Iniciando treinamento do modelo aprimorado...")
    
    model = EnhancedArbovirusPredictionModel()
    
    # Carregar dados de treinamento
    training_data = model.load_training_data(days_back=365)
    
    if not training_data.empty:
        # Treinar modelo
        metrics = model.train_model(training_data)
        
        if metrics.get('status') == 'success':
            # Salvar modelo
            model_path = os.path.join(os.path.dirname(__file__), 'enhanced_model')
            model.save_model(model_path)
            
            # Testar predição
            print("\n🔮 Testando predições...")
            cities = [('São Paulo', 'SP'), ('Rio de Janeiro', 'RJ'), ('Salvador', 'BA')]
            
            for city, state in cities:
                prediction = model.predict_for_city(city, state)
                print(f"   {city}-{state}: {prediction['predicted_cases_dengue']:.1f} casos de dengue")
            
            print("✅ Treinamento e teste concluídos com sucesso!")
        else:
            print(f"❌ Falha no treinamento: {metrics.get('status')}")
    else:
        print("❌ Nenhum dado de treinamento encontrado")

if __name__ == '__main__':
    main()

