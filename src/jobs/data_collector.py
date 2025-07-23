"""
Job de coleta diária de dados climáticos e de arboviroses
"""
import os
import sys
import requests
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask
from src.models.user import db
from src.models.climate import ClimateData
from src.models.arbovirus import ArbovirusData

class DataCollector:
    """Classe responsável pela coleta de dados de APIs externas"""
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        
    def create_app(self):
        """Criar aplicação Flask se não fornecida"""
        if not self.app:
            app = Flask(__name__)
            app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'database', 'app.db')}"
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)
            return app
        return self.app
    
    def collect_climate_data(self, cities: List[tuple]) -> List[Dict]:
        """
        Coletar dados climáticos para as cidades especificadas
        
        Args:
            cities: Lista de tuplas (cidade, estado)
            
        Returns:
            Lista de dicionários com dados climáticos
        """
        climate_data = []
        
        for city, state in cities:
            try:
                # Usar Open-Meteo API (gratuita, sem necessidade de API key)
                # Coordenadas aproximadas das principais cidades brasileiras
                coordinates = self._get_city_coordinates(city, state)
                
                if coordinates:
                    lat, lon = coordinates
                    url = f"https://api.open-meteo.com/v1/forecast"
                    params = {
                        'latitude': lat,
                        'longitude': lon,
                        'current': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m',
                        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
                        'timezone': 'America/Sao_Paulo',
                        'forecast_days': 1
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        current = data.get('current', {})
                        daily = data.get('daily', {})
                        
                        climate_record = {
                            'date': date.today(),
                            'city': city,
                            'state': state,
                            'temperature_avg': current.get('temperature_2m'),
                            'temperature_min': daily.get('temperature_2m_min', [None])[0],
                            'temperature_max': daily.get('temperature_2m_max', [None])[0],
                            'precipitation': daily.get('precipitation_sum', [None])[0],
                            'humidity_avg': current.get('relative_humidity_2m'),
                            'wind_speed_avg': current.get('wind_speed_10m')
                        }
                        
                        climate_data.append(climate_record)
                        print(f"✅ Dados climáticos coletados para {city}-{state}")
                    else:
                        print(f"❌ Erro ao coletar dados climáticos para {city}-{state}: {response.status_code}")
                        
            except Exception as e:
                print(f"❌ Erro ao processar {city}-{state}: {str(e)}")
                
        return climate_data
    
    def collect_arbovirus_data(self, cities: List[tuple]) -> List[Dict]:
        """
        Coletar dados de arboviroses do InfoDengue
        
        Args:
            cities: Lista de tuplas (cidade, estado)
            
        Returns:
            Lista de dicionários com dados de arboviroses
        """
        arbovirus_data = []
        
        # Mapear códigos IBGE das principais cidades
        city_codes = self._get_ibge_codes()
        
        for city, state in cities:
            try:
                city_key = f"{city}-{state}"
                ibge_code = city_codes.get(city_key)
                
                if ibge_code:
                    # API do InfoDengue direta
                    url = "https://info.dengue.mat.br/api/alertcity"
                    params = {
                        'geocode': ibge_code,
                        'disease': 'dengue',
                        'format': 'json',
                        'ew_start': 1,
                        'ew_end': 53,
                        'ey_start': 2024,
                        'ey_end': 2025
                    }
                    
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data and len(data) > 0:
                            # Pegar o registro mais recente
                            latest_record = data[-1]
                            
                            arbovirus_record = {
                                'date': date.today(),  # Usar data atual por simplicidade
                                'city': city,
                                'state': state,
                                'cases_dengue': latest_record.get('casos', 0),
                                'cases_zika': 0,  # InfoDengue foca principalmente em dengue
                                'cases_chikungunya': 0,
                                'incidence_dengue': latest_record.get('incidencia', 0),
                                'incidence_zika': 0,
                                'incidence_chikungunya': 0
                            }
                            
                            arbovirus_data.append(arbovirus_record)
                            print(f"✅ Dados de arboviroses coletados para {city}-{state}")
                        else:
                            print(f"⚠️ Nenhum dado de arboviroses encontrado para {city}-{state}")
                    else:
                        print(f"❌ Erro ao coletar dados de arboviroses para {city}-{state}: {response.status_code}")
                else:
                    print(f"⚠️ Código IBGE não encontrado para {city}-{state}")
                    
            except Exception as e:
                print(f"❌ Erro ao processar arboviroses para {city}-{state}: {str(e)}")
                
        return arbovirus_data
    
    def save_climate_data(self, climate_data: List[Dict]):
        """Salvar dados climáticos no banco de dados"""
        app = self.create_app()
        
        with app.app_context():
            for record in climate_data:
                # Verificar se já existe registro para a data e cidade
                existing = ClimateData.query.filter_by(
                    date=record['date'],
                    city=record['city'],
                    state=record['state']
                ).first()
                
                if not existing:
                    climate_obj = ClimateData(**record)
                    db.session.add(climate_obj)
                else:
                    # Atualizar registro existente
                    for key, value in record.items():
                        if key not in ['date', 'city', 'state']:
                            setattr(existing, key, value)
            
            db.session.commit()
            print(f"✅ {len(climate_data)} registros climáticos salvos no banco de dados")
    
    def save_arbovirus_data(self, arbovirus_data: List[Dict]):
        """Salvar dados de arboviroses no banco de dados"""
        app = self.create_app()
        
        with app.app_context():
            for record in arbovirus_data:
                # Verificar se já existe registro para a data e cidade
                existing = ArbovirusData.query.filter_by(
                    date=record['date'],
                    city=record['city'],
                    state=record['state']
                ).first()
                
                if not existing:
                    arbovirus_obj = ArbovirusData(**record)
                    db.session.add(arbovirus_obj)
                else:
                    # Atualizar registro existente
                    for key, value in record.items():
                        if key not in ['date', 'city', 'state']:
                            setattr(existing, key, value)
            
            db.session.commit()
            print(f"✅ {len(arbovirus_data)} registros de arboviroses salvos no banco de dados")
    
    def _get_city_coordinates(self, city: str, state: str) -> Optional[tuple]:
        """Obter coordenadas aproximadas das cidades"""
        coordinates = {
            'São Paulo-SP': (-23.5505, -46.6333),
            'Rio de Janeiro-RJ': (-22.9068, -43.1729),
            'Belo Horizonte-MG': (-19.9167, -43.9345),
            'Salvador-BA': (-12.9714, -38.5014),
            'Fortaleza-CE': (-3.7319, -38.5267),
            'Recife-PE': (-8.0476, -34.8770),
            'Manaus-AM': (-3.1190, -60.0217),
            'Brasília-DF': (-15.8267, -47.9218)
        }
        
        return coordinates.get(f"{city}-{state}")
    
    def _get_ibge_codes(self) -> Dict[str, str]:
        """Mapear códigos IBGE das principais cidades"""
        return {
            'São Paulo-SP': '3550308',
            'Rio de Janeiro-RJ': '3304557',
            'Belo Horizonte-MG': '3106200',
            'Salvador-BA': '2927408',
            'Fortaleza-CE': '2304400',
            'Recife-PE': '2611606',
            'Manaus-AM': '1302603',
            'Brasília-DF': '5300108'
        }
    
    def run_daily_collection(self):
        """Executar coleta diária de dados"""
        print(f"🚀 Iniciando coleta diária de dados - {datetime.now()}")
        
        # Cidades principais para coleta
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
            # Coletar dados climáticos
            print("📊 Coletando dados climáticos...")
            climate_data = self.collect_climate_data(cities)
            if climate_data:
                self.save_climate_data(climate_data)
            
            # Coletar dados de arboviroses
            print("🦟 Coletando dados de arboviroses...")
            arbovirus_data = self.collect_arbovirus_data(cities)
            if arbovirus_data:
                self.save_arbovirus_data(arbovirus_data)
            
            print(f"✅ Coleta diária concluída com sucesso - {datetime.now()}")
            
        except Exception as e:
            print(f"❌ Erro durante a coleta diária: {str(e)}")
            raise

def main():
    """Função principal para execução do job"""
    collector = DataCollector()
    collector.run_daily_collection()

if __name__ == '__main__':
    main()

