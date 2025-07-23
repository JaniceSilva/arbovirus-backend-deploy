"""
Script de migração e configuração do banco de dados
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask
from src.models.user import db, User
from src.models.climate import ClimateData
from src.models.arbovirus import ArbovirusData
from src.models.prediction import PredictionResult
from datetime import datetime, date, timedelta
import random

def create_app():
    """Criar aplicação Flask para migração"""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'app.db')}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app

def create_tables():
    """Criar todas as tabelas do banco de dados"""
    app = create_app()
    with app.app_context():
        db.create_all()
        print("✅ Tabelas criadas com sucesso!")

def drop_tables():
    """Remover todas as tabelas do banco de dados"""
    app = create_app()
    with app.app_context():
        db.drop_all()
        print("✅ Tabelas removidas com sucesso!")

def seed_sample_data():
    """Popular o banco com dados de exemplo"""
    app = create_app()
    with app.app_context():
        # Verificar se as tabelas existem antes de tentar limpar
        try:
            db.session.query(ClimateData).delete()
            db.session.query(ArbovirusData).delete()
            db.session.query(PredictionResult).delete()
            db.session.commit()
        except Exception as e:
            print(f"Aviso: Erro ao limpar dados existentes: {e}")
            db.session.rollback()
        
        # Cidades de exemplo
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
        
        # Gerar dados dos últimos 90 dias
        start_date = date.today() - timedelta(days=90)
        
        for i in range(90):
            current_date = start_date + timedelta(days=i)
            
            for city, state in cities:
                # Dados climáticos
                climate_data = ClimateData(
                    date=current_date,
                    city=city,
                    state=state,
                    temperature_avg=random.uniform(20, 35),
                    temperature_min=random.uniform(15, 25),
                    temperature_max=random.uniform(25, 40),
                    precipitation=random.uniform(0, 50),
                    humidity_avg=random.uniform(40, 90),
                    wind_speed_avg=random.uniform(5, 25)
                )
                db.session.add(climate_data)
                
                # Dados de arboviroses (apenas alguns dias com casos)
                if random.random() < 0.3:  # 30% de chance de ter casos
                    arbovirus_data = ArbovirusData(
                        date=current_date,
                        city=city,
                        state=state,
                        cases_dengue=random.randint(0, 50),
                        cases_zika=random.randint(0, 10),
                        cases_chikungunya=random.randint(0, 15),
                        incidence_dengue=random.uniform(0, 100),
                        incidence_zika=random.uniform(0, 20),
                        incidence_chikungunya=random.uniform(0, 30)
                    )
                    db.session.add(arbovirus_data)
        
        # Gerar algumas predições de exemplo
        prediction_date = date.today() + timedelta(days=30)
        for city, state in cities:
            prediction = PredictionResult(
                prediction_date=prediction_date,
                city=city,
                state=state,
                predicted_cases_dengue=random.uniform(10, 100),
                predicted_cases_zika=random.uniform(0, 20),
                predicted_cases_chikungunya=random.uniform(0, 30),
                model_version='v1.0.0',
                confidence_score=random.uniform(0.7, 0.95)
            )
            db.session.add(prediction)
        
        db.session.commit()
        print("✅ Dados de exemplo inseridos com sucesso!")

def reset_database():
    """Resetar completamente o banco de dados"""
    drop_tables()
    create_tables()
    seed_sample_data()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python migrations.py [create|drop|seed|reset]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'create':
        create_tables()
    elif command == 'drop':
        drop_tables()
    elif command == 'seed':
        seed_sample_data()
    elif command == 'reset':
        reset_database()
    else:
        print("Comando inválido. Use: create, drop, seed ou reset")
        sys.exit(1)

