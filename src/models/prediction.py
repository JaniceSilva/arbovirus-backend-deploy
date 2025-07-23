from src.models.user import db
from datetime import datetime

class PredictionResult(db.Model):
    __tablename__ = 'prediction_results'
    
    id = db.Column(db.Integer, primary_key=True)
    prediction_date = db.Column(db.Date, nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    predicted_cases_dengue = db.Column(db.Numeric(10, 4))
    predicted_cases_zika = db.Column(db.Numeric(10, 4))
    predicted_cases_chikungunya = db.Column(db.Numeric(10, 4))
    model_version = db.Column(db.String(50))
    confidence_score = db.Column(db.Numeric(5, 4))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PredictionResult {self.city}-{self.state} {self.prediction_date}>'

    def to_dict(self):
        return {
            'id': self.id,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'city': self.city,
            'state': self.state,
            'predicted_cases_dengue': float(self.predicted_cases_dengue) if self.predicted_cases_dengue else None,
            'predicted_cases_zika': float(self.predicted_cases_zika) if self.predicted_cases_zika else None,
            'predicted_cases_chikungunya': float(self.predicted_cases_chikungunya) if self.predicted_cases_chikungunya else None,
            'model_version': self.model_version,
            'confidence_score': float(self.confidence_score) if self.confidence_score else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

