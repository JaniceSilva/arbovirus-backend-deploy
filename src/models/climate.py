from src.models.user import db
from datetime import datetime

class ClimateData(db.Model):
    __tablename__ = 'climatic_data'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    temperature_avg = db.Column(db.Numeric(5, 2))
    temperature_min = db.Column(db.Numeric(5, 2))
    temperature_max = db.Column(db.Numeric(5, 2))
    precipitation = db.Column(db.Numeric(8, 2))
    humidity_avg = db.Column(db.Numeric(5, 2))
    wind_speed_avg = db.Column(db.Numeric(5, 2))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ClimateData {self.city}-{self.state} {self.date}>'

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'city': self.city,
            'state': self.state,
            'temperature_avg': float(self.temperature_avg) if self.temperature_avg else None,
            'temperature_min': float(self.temperature_min) if self.temperature_min else None,
            'temperature_max': float(self.temperature_max) if self.temperature_max else None,
            'precipitation': float(self.precipitation) if self.precipitation else None,
            'humidity_avg': float(self.humidity_avg) if self.humidity_avg else None,
            'wind_speed_avg': float(self.wind_speed_avg) if self.wind_speed_avg else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

