from flask import Blueprint, jsonify
from datetime import datetime

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde da aplicação"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'Arbovirus Prediction API',
        'version': '1.0.0'
    })

@health_bp.route('/status', methods=['GET'])
def status():
    """Endpoint de status detalhado da aplicação"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'Arbovirus Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'climate': '/api/climate',
            'arbovirus': '/api/arbovirus',
            'predictions': '/api/predictions',
            'health': '/api/health'
        }
    })

