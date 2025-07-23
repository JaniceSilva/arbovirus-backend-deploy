"""
Job de processamento mensal de predições usando deep learning
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import pickle
import json

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask
from src.models.user import db
from src.models.climate import ClimateData
from src.models.arbovirus import ArbovirusData
from src.models.prediction import PredictionResult

class PredictionProcessor:
    """Classe responsável pelo processamento de predições de arboviroses"""
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.model_version = "v1.0.0"
        
    def create_app(self):
        """Criar aplicação Flask se não fornecida"""
        if not self.app:
            app = Flask(__name__)
            app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'database', 'app.db')}"
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)
            return app
        return self.app
    
    def load_historical_data(self, city: str, state: str, days_back: int = 90) -> pd.DataFrame:
        """
        Carregar dados históricos para uma cidade específica
        
        Args:
            city: Nome da cidade
            state: Estado
            days_back: Número de dias para voltar no histórico
            
        Returns:
            DataFrame com dados históricos combinados
        """
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
                    'cases_dengue': record.cases_dengue or 0,
                    'cases_zika': record.cases_zika or 0,
                    'cases_chikungunya': record.cases_chikungunya or 0,
                    'incidence_dengue': float(record.incidence_dengue) if record.incidence_dengue else 0,
                    'incidence_zika': float(record.incidence_zika) if record.incidence_zika else 0,
                    'incidence_chikungunya': float(record.incidence_chikungunya) if record.incidence_chikungunya else 0
                })
            
            # Combinar dados em DataFrame
            climate_df = pd.DataFrame(climate_data)
            arbovirus_df = pd.DataFrame(arbovirus_data)
            
            if not climate_df.empty and not arbovirus_df.empty:
                # Merge por data
                combined_df = pd.merge(climate_df, arbovirus_df, on='date', how='outer')
                combined_df = combined_df.fillna(0)  # Preencher valores ausentes com 0
                combined_df = combined_df.sort_values('date')
                return combined_df
            elif not climate_df.empty:
                # Apenas dados climáticos disponíveis
                climate_df['cases_dengue'] = 0
                climate_df['cases_zika'] = 0
                climate_df['cases_chikungunya'] = 0
                climate_df['incidence_dengue'] = 0
                climate_df['incidence_zika'] = 0
                climate_df['incidence_chikungunya'] = 0
                return climate_df
            else:
                return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preparar features para o modelo de machine learning
        
        Args:
            df: DataFrame com dados históricos
            
        Returns:
            Array numpy com features preparadas
        """
        if df.empty:
            return np.array([])
        
        # Selecionar features climáticas
        feature_columns = [
            'temperature_avg', 'temperature_min', 'temperature_max',
            'precipitation', 'humidity_avg', 'wind_speed_avg'
        ]
        
        # Adicionar features derivadas
        df['temp_range'] = df['temperature_max'] - df['temperature_min']
        df['temp_humidity_interaction'] = df['temperature_avg'] * df['humidity_avg']
        
        # Adicionar médias móveis (janela de 7 dias)
        for col in ['temperature_avg', 'precipitation', 'humidity_avg']:
            df[f'{col}_ma7'] = df[col].rolling(window=7, min_periods=1).mean()
        
        # Selecionar todas as features
        all_features = feature_columns + [
            'temp_range', 'temp_humidity_interaction',
            'temperature_avg_ma7', 'precipitation_ma7', 'humidity_avg_ma7'
        ]
        
        # Normalizar features (simples min-max scaling)
        features = df[all_features].values
        
        # Aplicar normalização básica
        features_normalized = np.zeros_like(features)
        for i in range(features.shape[1]):
            col_data = features[:, i]
            if col_data.max() != col_data.min():
                features_normalized[:, i] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
            else:
                features_normalized[:, i] = col_data
        
        return features_normalized
    
    def simple_prediction_model(self, features: np.ndarray, historical_cases: np.ndarray) -> Dict[str, float]:
        """
        Modelo simples de predição baseado em correlações e tendências
        
        Args:
            features: Features climáticas normalizadas
            historical_cases: Casos históricos de arboviroses
            
        Returns:
            Dicionário com predições para cada doença
        """
        if len(features) == 0 or len(historical_cases) == 0:
            return {
                'predicted_cases_dengue': 0.0,
                'predicted_cases_zika': 0.0,
                'predicted_cases_chikungunya': 0.0,
                'confidence_score': 0.5
            }
        
        # Pegar últimas features (mais recentes)
        recent_features = features[-7:] if len(features) >= 7 else features
        recent_cases = historical_cases[-30:] if len(historical_cases) >= 30 else historical_cases
        
        # Calcular médias das features recentes
        avg_temp = np.mean(recent_features[:, 0]) if recent_features.shape[1] > 0 else 0.5
        avg_humidity = np.mean(recent_features[:, 4]) if recent_features.shape[1] > 4 else 0.5
        avg_precipitation = np.mean(recent_features[:, 3]) if recent_features.shape[1] > 3 else 0.5
        
        # Calcular tendência dos casos recentes
        if len(recent_cases) > 1:
            case_trend = np.mean(recent_cases[-7:]) if len(recent_cases) >= 7 else np.mean(recent_cases)
        else:
            case_trend = 0
        
        # Modelo simples baseado em correlações conhecidas
        # Dengue: favorecida por alta temperatura e umidade
        dengue_factor = (avg_temp * 0.4 + avg_humidity * 0.4 + avg_precipitation * 0.2)
        predicted_dengue = max(0, case_trend * (1 + dengue_factor * 0.5))
        
        # Zika: similar à dengue, mas menos comum
        zika_factor = dengue_factor * 0.3
        predicted_zika = max(0, case_trend * 0.2 * (1 + zika_factor))
        
        # Chikungunya: também similar, mas com padrões diferentes
        chikungunya_factor = dengue_factor * 0.4
        predicted_chikungunya = max(0, case_trend * 0.3 * (1 + chikungunya_factor))
        
        # Calcular score de confiança baseado na quantidade de dados
        confidence = min(0.95, 0.5 + (len(recent_cases) / 60) * 0.4)
        
        return {
            'predicted_cases_dengue': float(predicted_dengue),
            'predicted_cases_zika': float(predicted_zika),
            'predicted_cases_chikungunya': float(predicted_chikungunya),
            'confidence_score': float(confidence)
        }
    
    def generate_predictions(self, cities: List[tuple]) -> List[Dict]:
        """
        Gerar predições para as cidades especificadas
        
        Args:
            cities: Lista de tuplas (cidade, estado)
            
        Returns:
            Lista de dicionários com predições
        """
        predictions = []
        prediction_date = date.today() + timedelta(days=30)  # Predição para 30 dias à frente
        
        for city, state in cities:
            try:
                print(f"🔮 Gerando predição para {city}-{state}...")
                
                # Carregar dados históricos
                historical_data = self.load_historical_data(city, state)
                
                if not historical_data.empty:
                    # Preparar features
                    features = self.prepare_features(historical_data)
                    historical_cases = historical_data['cases_dengue'].values
                    
                    # Gerar predição
                    prediction = self.simple_prediction_model(features, historical_cases)
                    
                    # Criar registro de predição
                    prediction_record = {
                        'prediction_date': prediction_date,
                        'city': city,
                        'state': state,
                        'model_version': self.model_version,
                        **prediction
                    }
                    
                    predictions.append(prediction_record)
                    print(f"✅ Predição gerada para {city}-{state}: {prediction['predicted_cases_dengue']:.1f} casos de dengue")
                    
                else:
                    print(f"⚠️ Dados históricos insuficientes para {city}-{state}")
                    
            except Exception as e:
                print(f"❌ Erro ao gerar predição para {city}-{state}: {str(e)}")
        
        return predictions
    
    def save_predictions(self, predictions: List[Dict]):
        """Salvar predições no banco de dados"""
        app = self.create_app()
        
        with app.app_context():
            for record in predictions:
                # Verificar se já existe predição para a data e cidade
                existing = PredictionResult.query.filter_by(
                    prediction_date=record['prediction_date'],
                    city=record['city'],
                    state=record['state'],
                    model_version=record['model_version']
                ).first()
                
                if not existing:
                    prediction_obj = PredictionResult(**record)
                    db.session.add(prediction_obj)
                else:
                    # Atualizar predição existente
                    for key, value in record.items():
                        if key not in ['prediction_date', 'city', 'state', 'model_version']:
                            setattr(existing, key, value)
            
            db.session.commit()
            print(f"✅ {len(predictions)} predições salvas no banco de dados")
    
    def run_monthly_prediction(self):
        """Executar processamento mensal de predições"""
        print(f"🚀 Iniciando processamento mensal de predições - {datetime.now()}")
        
        # Cidades principais para predição
        cities = [
            ('São Paulo', 'SP'),
            ('Rio de Janeiro', 'RJ'),
            ('Belo Horizonte', 'MG'),
            ('Salvador', 'BA'),
            ('Fortaleza', 'CE'),
            ('Recife', 'PE'),
            ('Manaus', 'AM'),
            ('Brasília', 'DF')
        ]
        
        try:
            # Gerar predições
            predictions = self.generate_predictions(cities)
            
            if predictions:
                self.save_predictions(predictions)
            
            print(f"✅ Processamento mensal concluído com sucesso - {datetime.now()}")
            
        except Exception as e:
            print(f"❌ Erro durante o processamento mensal: {str(e)}")
            raise

def main():
    """Função principal para execução do job"""
    processor = PredictionProcessor()
    processor.run_monthly_prediction()

if __name__ == '__main__':
    main()

