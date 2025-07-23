from flask import Blueprint, jsonify, request
from src.models.arbovirus import ArbovirusData, db
from datetime import datetime

arbovirus_bp = Blueprint('arbovirus', __name__)

@arbovirus_bp.route('/arbovirus', methods=['GET'])
def get_arbovirus_data():
    """Obter dados de arboviroses com filtros opcionais"""
    city = request.args.get('city')
    state = request.args.get('state')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    disease = request.args.get('disease')  # dengue, zika, chikungunya
    
    query = ArbovirusData.query
    
    if city:
        query = query.filter(ArbovirusData.city.ilike(f'%{city}%'))
    if state:
        query = query.filter(ArbovirusData.state.ilike(f'%{state}%'))
    if start_date:
        query = query.filter(ArbovirusData.date >= datetime.strptime(start_date, '%Y-%m-%d').date())
    if end_date:
        query = query.filter(ArbovirusData.date <= datetime.strptime(end_date, '%Y-%m-%d').date())
    
    arbovirus_data = query.order_by(ArbovirusData.date.desc()).limit(100).all()
    
    # Filtrar por doença específica se solicitado
    if disease:
        filtered_data = []
        for data in arbovirus_data:
            data_dict = data.to_dict()
            if disease == 'dengue':
                data_dict = {k: v for k, v in data_dict.items() if 'dengue' in k or k in ['id', 'date', 'city', 'state', 'created_at']}
            elif disease == 'zika':
                data_dict = {k: v for k, v in data_dict.items() if 'zika' in k or k in ['id', 'date', 'city', 'state', 'created_at']}
            elif disease == 'chikungunya':
                data_dict = {k: v for k, v in data_dict.items() if 'chikungunya' in k or k in ['id', 'date', 'city', 'state', 'created_at']}
            filtered_data.append(data_dict)
        return jsonify(filtered_data)
    
    return jsonify([data.to_dict() for data in arbovirus_data])

@arbovirus_bp.route('/arbovirus', methods=['POST'])
def create_arbovirus_data():
    """Criar novo registro de dados de arboviroses"""
    data = request.json
    
    arbovirus_data = ArbovirusData(
        date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
        city=data['city'],
        state=data['state'],
        cases_dengue=data.get('cases_dengue', 0),
        cases_zika=data.get('cases_zika', 0),
        cases_chikungunya=data.get('cases_chikungunya', 0),
        incidence_dengue=data.get('incidence_dengue'),
        incidence_zika=data.get('incidence_zika'),
        incidence_chikungunya=data.get('incidence_chikungunya')
    )
    
    db.session.add(arbovirus_data)
    db.session.commit()
    return jsonify(arbovirus_data.to_dict()), 201

@arbovirus_bp.route('/arbovirus/<int:arbovirus_id>', methods=['GET'])
def get_arbovirus_data_by_id(arbovirus_id):
    """Obter dados de arboviroses por ID"""
    arbovirus_data = ArbovirusData.query.get_or_404(arbovirus_id)
    return jsonify(arbovirus_data.to_dict())

@arbovirus_bp.route('/arbovirus/summary', methods=['GET'])
def get_arbovirus_summary():
    """Obter resumo dos dados de arboviroses por cidade/estado"""
    city = request.args.get('city')
    state = request.args.get('state')
    
    query = db.session.query(
        ArbovirusData.city,
        ArbovirusData.state,
        db.func.sum(ArbovirusData.cases_dengue).label('total_dengue'),
        db.func.sum(ArbovirusData.cases_zika).label('total_zika'),
        db.func.sum(ArbovirusData.cases_chikungunya).label('total_chikungunya'),
        db.func.count(ArbovirusData.id).label('total_records')
    )
    
    if city:
        query = query.filter(ArbovirusData.city.ilike(f'%{city}%'))
    if state:
        query = query.filter(ArbovirusData.state.ilike(f'%{state}%'))
    
    summary = query.group_by(ArbovirusData.city, ArbovirusData.state).all()
    
    return jsonify([{
        'city': row.city,
        'state': row.state,
        'total_dengue': row.total_dengue or 0,
        'total_zika': row.total_zika or 0,
        'total_chikungunya': row.total_chikungunya or 0,
        'total_records': row.total_records
    } for row in summary])

