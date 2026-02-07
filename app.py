from flask import Flask, render_template, redirect, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    text = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Article %r>' % self.id


with app.app_context():
    db.create_all()


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route("/create-article", methods=['POST', 'GET'])
def create_article():
    if request.method == "POST":
        text = request.form['text']

        article = Article(text=text)

        try:
            db.session.add(article)
            db.session.commit()
            return redirect('/ai')  # После создания → на список статей
        except:
            return "При добавлении статьи произошла ошибка"

    else:  # GET запрос
        # Показываем форму создания
        return render_template("create_article.html")  # ← Новый шаблон


@app.route("/ai")
def posts():
    articles = Article.query.order_by(Article.date.desc()).all()
    return render_template("ai.html", articles=articles)


if __name__ == '__main__':
    app.run(debug=True)