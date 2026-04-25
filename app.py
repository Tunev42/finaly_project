from flask import Flask, render_template, request, redirect, session, flash, jsonify
from models import db, User, Verification, Premium
from payments import create_premium_payment, check_payment
from factcheck import check_text
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'somesecretkey2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///factchecker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

DAILY_LIMIT = 5


def get_remaining(username):
    prem = Premium.query.get(username)
    if prem and prem.until > datetime.utcnow():
        return 999

    today = datetime.utcnow().date()
    used = Verification.query.filter(
        Verification.username == username,
        Verification.created >= today
    ).count()
    return max(0, DAILY_LIMIT - used)


@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/auto', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['username'].strip()
        pwd = request.form['password']
        email = request.form.get('email', '')

        if len(name) < 3:
            flash('Логин слишком короткий')
        elif User.query.get(name):
            flash('Такой логин уже есть')
        elif len(pwd) < 6:
            flash('Пароль минимум 6 символов')
        else:
            user = User(username=name, email=email)
            user.set_password(pwd)
            db.session.add(user)
            db.session.commit()
            flash('Регистрация прошла, теперь войдите')
            return redirect('/login')
    return render_template('auto.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.get(request.form['username'])
        if user and user.check_password(request.form['password']):
            session['user'] = user.username
            session['logged_in'] = True
            flash(f'Привет, {user.username}')
            return redirect('/dashboard')
        flash('Неверный логин или пароль')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    username = session['user']
    remaining = get_remaining(username)
    is_premium = remaining == 999
    total = Verification.query.filter_by(username=username).count()
    today = datetime.utcnow().date()
    today_count = Verification.query.filter(
        Verification.username == username,
        Verification.created >= today
    ).count()
    return render_template('dashboard.html',
        username=username,
        remaining=remaining,
        is_premium=is_premium,
        total_checks=total,
        today_checks=today_count
    )


@app.route('/ai', methods=['GET', 'POST'])
def ai():
    if 'user' not in session:
        return redirect('/login')
    username = session['user']
    remaining = get_remaining(username)
    is_premium = remaining == 999
    result = None
    error = None

    if request.method == 'POST':
        text = request.form.get('title', '').strip()
        if len(text) < 10:
            flash('Слишком короткий текст')
        elif not is_premium and remaining <= 0:
            flash('Закончились проверки на сегодня')
        else:
            verdict, sources = check_text(text)
            v = Verification(username=username, text=text[:500], verdict=verdict)
            db.session.add(v)
            db.session.commit()
            result = f'{verdict}\n\nИсточники: ' + ', '.join(sources[:3])
            remaining = get_remaining(username)

    return render_template('ai.html',
        result=result,
        error=error,
        tokens_remaining=remaining,
        is_premium=is_premium,
        username=username
    )


@app.route('/upgrade')
def upgrade():
    if 'user' not in session:
        return redirect('/login')
    username = session['user']
    remaining = get_remaining(username)
    is_premium = remaining == 999
    return render_template('upgrade.html',
        remaining=remaining,
        is_premium=is_premium,
        username=username
    )


@app.route('/create-payment', methods=['POST'])
def create_payment_route():
    if 'user' not in session:
        return jsonify({'error': 'no auth'}), 401
    data = request.get_json()
    plan = data.get('plan')
    username = session['user']
    if plan == 'month':
        price = 299
        days = 30
    elif plan == 'year':
        price = 1990
        days = 365
    else:
        return jsonify({'error': 'bad plan'}), 400
    payment_id, url = create_premium_payment(username, price, days)
    return jsonify({'payment_url': url, 'payment_id': payment_id})


@app.route('/payment/success')
def payment_success():
    pid = request.args.get('payment_id')
    if not pid:
        return redirect('/upgrade')
    if check_payment(pid):
        flash('Премиум активирован')
    else:
        flash('Ошибка оплаты')
    return redirect('/dashboard')


@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')
    checks = Verification.query.filter_by(username=session['user']).order_by(Verification.created.desc()).limit(30).all()
    return render_template('history.html', verifications=checks)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.get('admin'):
            admin = User(username='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
    app.run(debug=True)
