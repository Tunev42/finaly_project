from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'
    username = db.Column(db.String(50), primary_key=True)
    password_hash = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(100))
    created = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pwd):
        self.password_hash = generate_password_hash(pwd)

    def check_password(self, pwd):
        return check_password_hash(self.password_hash, pwd)


class Verification(db.Model):
    __tablename__ = 'verifications'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), db.ForeignKey('users.username'))
    text = db.Column(db.String(1000))
    verdict = db.Column(db.String(100))
    created = db.Column(db.DateTime, default=datetime.utcnow)


class Premium(db.Model):
    __tablename__ = 'premiums'
    username = db.Column(db.String(50), primary_key=True)
    until = db.Column(db.DateTime)
    paid_at = db.Column(db.DateTime, default=datetime.utcnow)