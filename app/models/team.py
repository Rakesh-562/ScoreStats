from datetime import datetime
from app.extensions import db
class Team(db.Model):
    """
    Represent the Team 
    A team has multiple players and participates in matches
    """
    __tablename__='team'
    id=db.Column(db.Integer,primary_key=True)
    # Info
    name=db.Column(db.String(100),unique=True,nullable=False,index=True,comment="Full Team Name")
    short_name=db.Column(db.String(10),unique=True,nullable=False,comment="Abbreviated name")
    logo_url=db.Column(db.String(255),comment="URL to Logo")
    created_at=db.Column(db.DateTime,default=datetime.utcnow,nullable=False,comment="Record Creation Timestamp")
    updated_at=db.Column(db.DateTime,default=datetime.utcnow,onupdate=datetime.utcnow,nullable=False,comment="Record Update Timestamp")
    # Relationships
    players=db.relationship("Player",backref="team",lazy='dynamic',cascade='all,delete-orphan',order_by='Player.name')
    def __repr__(self):
        return f"<Team {self.name} ({self.short_name})>"
    def to_dict(self):
        """Convert Team object to dictionary"""
        return {
            'id':self.id,
            'name':self.name,
            'short_name':self.short_name,
            'logo_url':self.logo_url,
            'created_at':self.created_at.isoformat(),
            'updated_at':self.updated_at.isoformat()
        }