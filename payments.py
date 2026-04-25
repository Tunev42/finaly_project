import uuid
from datetime import datetime, timedelta
from models import db, Premium


def create_premium_payment(username, price_rub, days):
    pid = f"pay_{username}_{uuid.uuid4().hex[:6]}"
    url = f"/payment/success?payment_id={pid}"
    return pid, url


def check_payment(payment_id):
    try:
        username = payment_id.split('_')[1]
    except:
        return False

    prem = Premium.query.get(username)
    if prem:
        prem.until = datetime.utcnow() + timedelta(days=30)
    else:
        prem = Premium(username=username, until=datetime.utcnow() + timedelta(days=30))
        db.session.add(prem)
    db.session.commit()
    return True