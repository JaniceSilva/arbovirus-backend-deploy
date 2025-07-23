from flask import Blueprint, jsonify, request
from src.models.climate import ClimateData, db
from datetime import datetime

climate_bp = Blueprint('climate', __name__)

@climate_bp.route('/climate', methods=['GET'])
def get_climate_data():
    """Obter dados climáticos com filtros opcionais"""
    city = request.args.get('city')
    state = request.args.get('state')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    query = ClimateData.query
    
    if city:
        query = query.filter(ClimateData.city.ilike(f'%{city}%'))
    if state:
        query = query.filter(ClimateData.state.ilike(f'%{state}%'))
    if start_date:
        query = query.filter(ClimateData.date >= datetime.strptime(start_date, '%Y-%m-%d').date())
    if end_date:
        query = query.filter(ClimateData.date <= datetime.strptime(end_date, '%Y-%m-%d').date())
    
    climate_data = query.order_by(ClimateData.date.desc()).limit(100).all()
    return jsonify([data.to_dict() for data in climate_data])

@climate_bp.route('/climate', methods=['POST'])
def create_climate_data():
    """Criar novo registro de dados climáticos"""
    data = request.json
    
    climate_data = ClimateData(
        date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
        city=data['city'],
        state=data['state'],
        temperature_avg=data.get('temperature_avg'),
        temperature_min=data.get('temperature_min'),
        temperature_max=data.get('temperature_max'),
        precipitation=data.get('precipitation'),
        humidity_avg=data.get('humidity_avg'),
        wind_speed_avg=data.get('wind_speed_avg')
    )
    
    db.session.add(climate_data)
    db.session.commit()
    return jsonify(climate_data.to_dict()), 201

@climate_bp.route('/climate/<int:climate_id>', methods=['GET'])
def get_climate_data_by_id(climate_id):
    """Obter dados climáticos por ID"""
    climate_data = ClimateData.query.get_or_404(climate_id)
    return jsonify(climate_data.to_dict())

@climate_bp.route('/climate/cities', methods=['GET'])
def get_cities():
    """Obter lista de cidades disponíveis"""
    cities = db.session.query(ClimateData.city, ClimateData.state).distinct().all()
    return jsonify([{'city': city, 'state': state} for city, state in cities])

