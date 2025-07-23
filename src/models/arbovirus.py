from src.models.user import db
from datetime import datetime

class ArbovirusData(db.Model):
    __tablename__ = 'arbovirus_data'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    cases_dengue = db.Column(db.Integer, default=0)
    cases_zika = db.Column(db.Integer, default=0)
    cases_chikungunya = db.Column(db.Integer, default=0)
    incidence_dengue = db.Column(db.Numeric(10, 4))
    incidence_zika = db.Column(db.Numeric(10, 4))
    incidence_chikungunya = db.Column(db.Numeric(10, 4))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ArbovirusData {self.city}-{self.state} {self.date}>'

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'city': self.city,
            'state': self.state,
            'cases_dengue': self.cases_dengue,
            'cases_zika': self.cases_zika,
            'cases_chikungunya': self.cases_chikungunya,
            'incidence_dengue': float(self.incidence_dengue) if self.incidence_dengue else None,
            'incidence_zika': float(self.incidence_zika) if self.incidence_zika else None,
            'incidence_chikungunya': float(self.incidence_chikungunya) if self.incidence_chikungunya else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

