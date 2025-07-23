"""
Modelo de Deep Learning para predição de arboviroses
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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow não disponível. Usando modelo simplificado.")
    TENSORFLOW_AVAILABLE = False
    # Definir classes dummy para evitar erros de sintaxe
    class keras:
        class Model:
            pass
    StandardScaler = None
    MinMaxScaler = None

from flask import Flask
from src.models.user import db
from src.models.climate import ClimateData
from src.models.arbovirus import ArbovirusData
from src.models.prediction import PredictionResult

class ArbovirusDeepLearningModel:
    """Modelo de Deep Learning para predição de arboviroses"""
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.model = None
        if TENSORFLOW_AVAILABLE:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            self.scaler_features = StandardScaler()
            self.scaler_target = MinMaxScaler()
        else:
            self.scaler_features = None
            self.scaler_target = None
        self.model_version = "v2.0.0"
        self.sequence_length = 14  # Usar 14 dias de histórico para predição
        self.feature_names = [
            'temperature_avg', 'temperature_min', 'temperature_max',
            'precipitation', 'humidity_avg', 'wind_speed_avg',
            'temp_range', 'temp_humidity_interaction',
            'temperature_avg_ma7', 'precipitation_ma7', 'humidity_avg_ma7'
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
        
        # Criar features derivadas
        df['temp_range'] = df['temperature_max'] - df['temperature_min']
        df['temp_humidity_interaction'] = df['temperature_avg'] * df['humidity_avg']
        
        # Adicionar médias móveis por cidade
        for city_state in df[['city', 'state']].drop_duplicates().values:
            city, state = city_state
            mask = (df['city'] == city) & (df['state'] == state)
            
            for col in ['temperature_avg', 'precipitation', 'humidity_avg']:
                df.loc[mask, f'{col}_ma7'] = df.loc[mask, col].rolling(window=7, min_periods=1).mean()
        
        return df
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Criar sequências temporais para o modelo LSTM
        
        Args:
            data: Array de features
            target: Array de targets
            sequence_length: Comprimento da sequência
            
        Returns:
            Tupla com sequências de features e targets
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Construir modelo de deep learning
        
        Args:
            input_shape: Forma dos dados de entrada (sequence_length, n_features)
            
        Returns:
            Modelo Keras compilado
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível")
        
        model = keras.Sequential([
            # Camada LSTM para capturar padrões temporais
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            
            # Segunda camada LSTM
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Camadas densas para predição
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')  # Predição de casos de dengue
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Treinar o modelo de deep learning
        
        Args:
            df: DataFrame com dados de treinamento
            
        Returns:
            Métricas de treinamento
        """
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow não disponível. Usando modelo simplificado.")
            return {'status': 'tensorflow_not_available'}
        
        if df.empty:
            print("❌ Dados de treinamento vazios")
            return {'status': 'no_data'}
        
        print(f"📊 Preparando dados de treinamento com {len(df)} registros...")
        
        # Preparar features
        df = self.prepare_features(df)
        
        # Selecionar features e target
        feature_data = df[self.feature_names].values
        target_data = df['cases_dengue'].values
        
        # Normalizar dados
        feature_data_scaled = self.scaler_features.fit_transform(feature_data)
        target_data_scaled = self.scaler_target.fit_transform(target_data.reshape(-1, 1)).flatten()
        
        # Criar sequências temporais
        X, y = self.create_sequences(feature_data_scaled, target_data_scaled, self.sequence_length)
        
        if len(X) == 0:
            print("❌ Não foi possível criar sequências suficientes")
            return {'status': 'insufficient_sequences'}
        
        print(f"📈 Criadas {len(X)} sequências de treinamento")
        
        # Dividir dados em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Construir modelo
        self.model = self.build_model((self.sequence_length, len(self.feature_names)))
        
        print("🧠 Treinando modelo de deep learning...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Treinar modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Avaliar modelo
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Predições para métricas
        y_pred_train = self.model.predict(X_train, verbose=0)
        y_pred_val = self.model.predict(X_val, verbose=0)
        
        # Desnormalizar para calcular métricas reais
        y_train_real = self.scaler_target.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_pred_train_real = self.scaler_target.inverse_transform(y_pred_train).flatten()
        y_val_real = self.scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_val_real = self.scaler_target.inverse_transform(y_pred_val).flatten()
        
        metrics = {
            'train_loss': float(train_loss[0]),
            'val_loss': float(val_loss[0]),
            'train_mae': float(mean_absolute_error(y_train_real, y_pred_train_real)),
            'val_mae': float(mean_absolute_error(y_val_real, y_pred_val_real)),
            'train_r2': float(r2_score(y_train_real, y_pred_train_real)),
            'val_r2': float(r2_score(y_val_real, y_pred_val_real)),
            'status': 'success'
        }
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"   - MAE Validação: {metrics['val_mae']:.2f}")
        print(f"   - R² Validação: {metrics['val_r2']:.3f}")
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Fazer predição usando o modelo treinado
        
        Args:
            features: Array de features para predição
            
        Returns:
            Dicionário com predições
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            # Fallback para modelo simples
            return self._simple_prediction(features)
        
        try:
            # Normalizar features
            features_scaled = self.scaler_features.transform(features)
            
            # Criar sequência se necessário
            if len(features_scaled) >= self.sequence_length:
                sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            else:
                # Preencher com zeros se não houver dados suficientes
                padded = np.zeros((self.sequence_length, features_scaled.shape[1]))
                padded[-len(features_scaled):] = features_scaled
                sequence = padded.reshape(1, self.sequence_length, -1)
            
            # Fazer predição
            prediction_scaled = self.model.predict(sequence, verbose=0)[0][0]
            prediction = self.scaler_target.inverse_transform([[prediction_scaled]])[0][0]
            
            # Garantir que a predição seja não-negativa
            prediction = max(0, prediction)
            
            return {
                'predicted_cases_dengue': float(prediction),
                'predicted_cases_zika': float(prediction * 0.2),  # Estimativa baseada em proporção
                'predicted_cases_chikungunya': float(prediction * 0.3),
                'confidence_score': 0.85,  # Score alto para modelo treinado
                'model_type': 'deep_learning'
            }
            
        except Exception as e:
            print(f"❌ Erro na predição: {str(e)}")
            return self._simple_prediction(features)
    
    def _simple_prediction(self, features: np.ndarray) -> Dict[str, float]:
        """Modelo de predição simples como fallback"""
        if len(features) == 0:
            return {
                'predicted_cases_dengue': 0.0,
                'predicted_cases_zika': 0.0,
                'predicted_cases_chikungunya': 0.0,
                'confidence_score': 0.5,
                'model_type': 'simple'
            }
        
        # Usar últimas features
        recent_features = features[-7:] if len(features) >= 7 else features
        
        # Calcular médias
        avg_temp = np.mean(recent_features[:, 0]) if recent_features.shape[1] > 0 else 25
        avg_humidity = np.mean(recent_features[:, 4]) if recent_features.shape[1] > 4 else 70
        avg_precipitation = np.mean(recent_features[:, 3]) if recent_features.shape[1] > 3 else 5
        
        # Modelo baseado em correlações
        dengue_factor = (avg_temp / 35) * (avg_humidity / 100) * (1 + avg_precipitation / 50)
        predicted_dengue = max(0, dengue_factor * 50)
        
        return {
            'predicted_cases_dengue': float(predicted_dengue),
            'predicted_cases_zika': float(predicted_dengue * 0.2),
            'predicted_cases_chikungunya': float(predicted_dengue * 0.3),
            'confidence_score': 0.6,
            'model_type': 'simple'
        }
    
    def save_model(self, filepath: str):
        """Salvar modelo treinado"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            print("⚠️ Modelo não disponível para salvar")
            return
        
        try:
            # Salvar modelo Keras
            self.model.save(f"{filepath}_model.h5")
            
            # Salvar scalers
            with open(f"{filepath}_scaler_features.pkl", 'wb') as f:
                pickle.dump(self.scaler_features, f)
            
            with open(f"{filepath}_scaler_target.pkl", 'wb') as f:
                pickle.dump(self.scaler_target, f)
            
            # Salvar metadados
            metadata = {
                'model_version': self.model_version,
                'sequence_length': self.sequence_length,
                'feature_names': self.feature_names,
                'created_at': datetime.now().isoformat()
            }
            
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo salvo em {filepath}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {str(e)}")
    
    def load_model(self, filepath: str):
        """Carregar modelo treinado"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow não disponível")
            return
        
        try:
            # Carregar modelo Keras
            self.model = keras.models.load_model(f"{filepath}_model.h5")
            
            # Carregar scalers
            with open(f"{filepath}_scaler_features.pkl", 'rb') as f:
                self.scaler_features = pickle.load(f)
            
            with open(f"{filepath}_scaler_target.pkl", 'rb') as f:
                self.scaler_target = pickle.load(f)
            
            # Carregar metadados
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
                self.model_version = metadata.get('model_version', 'v2.0.0')
                self.sequence_length = metadata.get('sequence_length', 14)
                self.feature_names = metadata.get('feature_names', self.feature_names)
            
            print(f"✅ Modelo carregado de {filepath}")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {str(e)}")

def main():
    """Função principal para treinar o modelo"""
    print("🚀 Iniciando treinamento do modelo de deep learning...")
    
    model = ArbovirusDeepLearningModel()
    
    # Carregar dados de treinamento
    training_data = model.load_training_data(days_back=365)
    
    if not training_data.empty:
        # Treinar modelo
        metrics = model.train_model(training_data)
        
        if metrics.get('status') == 'success':
            # Salvar modelo
            model_path = os.path.join(os.path.dirname(__file__), 'trained_model')
            model.save_model(model_path)
            
            print("✅ Treinamento concluído com sucesso!")
        else:
            print(f"❌ Falha no treinamento: {metrics.get('status')}")
    else:
        print("❌ Nenhum dado de treinamento encontrado")

if __name__ == '__main__':
    main()

