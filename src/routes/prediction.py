from flask import Blueprint, jsonify, request
from src.models.prediction import PredictionResult, db
from datetime import datetime

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predictions', methods=['GET'])
def get_predictions():
    """Obter predições com filtros opcionais"""
    city = request.args.get('city')
    state = request.args.get('state')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    model_version = request.args.get('model_version')
    
    query = PredictionResult.query
    
    if city:
        query = query.filter(PredictionResult.city.ilike(f'%{city}%'))
    if state:
        query = query.filter(PredictionResult.state.ilike(f'%{state}%'))
    if start_date:
        query = query.filter(PredictionResult.prediction_date >= datetime.strptime(start_date, '%Y-%m-%d').date())
    if end_date:
        query = query.filter(PredictionResult.prediction_date <= datetime.strptime(end_date, '%Y-%m-%d').date())
    if model_version:
        query = query.filter(PredictionResult.model_version == model_version)
    
    predictions = query.order_by(PredictionResult.prediction_date.desc()).limit(100).all()
    return jsonify([prediction.to_dict() for prediction in predictions])

@prediction_bp.route('/predictions', methods=['POST'])
def create_prediction():
    """Criar nova predição"""
    data = request.json
    
    prediction = PredictionResult(
        prediction_date=datetime.strptime(data['prediction_date'], '%Y-%m-%d').date(),
        city=data['city'],
        state=data['state'],
        predicted_cases_dengue=data.get('predicted_cases_dengue'),
        predicted_cases_zika=data.get('predicted_cases_zika'),
        predicted_cases_chikungunya=data.get('predicted_cases_chikungunya'),
        model_version=data.get('model_version'),
        confidence_score=data.get('confidence_score')
    )
    
    db.session.add(prediction)
    db.session.commit()
    return jsonify(prediction.to_dict()), 201

@prediction_bp.route('/predictions/<int:prediction_id>', methods=['GET'])
def get_prediction_by_id(prediction_id):
    """Obter predição por ID"""
    prediction = PredictionResult.query.get_or_404(prediction_id)
    return jsonify(prediction.to_dict())

@prediction_bp.route('/predictions/latest', methods=['GET'])
def get_latest_predictions():
    """Obter as predições mais recentes por cidade"""
    city = request.args.get('city')
    state = request.args.get('state')
    
    # Subquery para encontrar a data mais recente de predição para cada cidade
    subquery = db.session.query(
        PredictionResult.city,
        PredictionResult.state,
        db.func.max(PredictionResult.prediction_date).label('max_date')
    )
    
    if city:
        subquery = subquery.filter(PredictionResult.city.ilike(f'%{city}%'))
    if state:
        subquery = subquery.filter(PredictionResult.state.ilike(f'%{state}%'))
    
    subquery = subquery.group_by(PredictionResult.city, PredictionResult.state).subquery()
    
    # Query principal para obter as predições mais recentes
    query = db.session.query(PredictionResult).join(
        subquery,
        (PredictionResult.city == subquery.c.city) &
        (PredictionResult.state == subquery.c.state) &
        (PredictionResult.prediction_date == subquery.c.max_date)
    )
    
    latest_predictions = query.all()
    return jsonify([prediction.to_dict() for prediction in latest_predictions])

@prediction_bp.route('/predictions/models', methods=['GET'])
def get_model_versions():
    """Obter lista de versões de modelos disponíveis"""
    models = db.session.query(PredictionResult.model_version).distinct().all()
    return jsonify([model[0] for model in models if model[0]])

