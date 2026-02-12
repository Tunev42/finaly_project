from flask import Flask, render_template, redirect, request, url_for, jsonify
import requests
import json
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import re
import torch
import urllib.parse
from bs4 import BeautifulSoup

load_dotenv()


fact_checker = None
similarity_model = None

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Article %r>' % self.id


with app.app_context():
    db.create_all()


def initialize_models():
    
    global fact_checker, similarity_model

    if fact_checker is None:
        try:
            fact_checker = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            print("Модель zero-shot-classification загружена")
        except Exception as e:
            print(f"Ошибка загрузки zero-shot-classification: {e}")
            fact_checker = None

    if similarity_model is None:
        try:
            similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Модель SentenceTransformer загружена")
        except Exception as e:
            print(f"Ошибка загрузки SentenceTransformer: {e}")
            similarity_model = None

    return fact_checker, similarity_model


def claims_from_text(text):
    
    if not text:
        return []

    # Разбиваем текст на предложения
    sentences = re.split(r'[.!?]+', text)

    # Фильтруем предложения
    claims = []
    for s in sentences:
        s = s.strip()
        # Оставляем предложения не больше 3 слов и не длиные
        if len(s.split()) > 3 and len(s) < 500 and len(s) > 20:
            claims.append(s)

    return claims[:5]  # Возвращаем не более 5 утверждений


def fetch_article_info_simple(url):
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Удаление скриптов и стилей
        for script in soup(["script", "style"]):
            script.decompose()

        # Получение заголовка
        title = soup.find('title')
        if title:
            title = title.get_text().strip()
        else:
            title = "Не удалось определить заголовок"

        # Получение основного текста
        text = soup.get_text()

        # Очистка текста
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return {
            "title": title,
            "text": text[:3000],  # Ограничиваем длину
            "success": True
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Превышено время ожидания ответа от сервера"
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Ошибка при загрузке страницы: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Неизвестная ошибка: {str(e)}"
        }


def analyze_claim_with_ai(claim):
    
    fact_checker_model, sim_model = initialize_models()

    # Проверка загружены ли модели
    if fact_checker_model is None:
        return {
            "claim": claim,
            "error": "Модель классификации не загружена",
            "category": "ошибка",
            "confidence": 0,
            "response": "Не удалось загрузить модель AI",
            "similar_facts": []
        }

    # Категории для классификации
    categories = [
        "правдивый факт",
        "ложная информация",
        "вводящее в заблуждение утверждение",
        "непроверенное утверждение",
        "частично правдиво"
    ]

    try:
        result = fact_checker_model(claim, categories)
        best_category = result['labels'][0]
        confidence = result['scores'][0]
    except Exception as e:
        return {
            "claim": claim,
            "error": f"Ошибка при анализе: {str(e)}",
            "category": "ошибка",
            "confidence": 0,
            "response": "Не удалось проанализировать утверждение",
            "similar_facts": []
        }

    responses = {
        "правдивый факт": " Это утверждение, вероятно, соответствует действительности.",
        "ложная информация": " Это утверждение, вероятно, содержит ложную информацию.",
        "вводящее в заблуждение утверждение": " Это утверждение может вводить в заблуждение.",
        "непроверенное утверждение": " Это утверждение требует дополнительной проверки.",
        "частично правдиво": " Это утверждение частично верно, но требует уточнений."
    }

    response_text = responses.get(best_category, " Требуется дополнительная проверка.")

    # Демо факты для сравнения
    demo_facts = {
        "Земля плоская": {"truth": False, "explanation": "Научно доказано, что Земля имеет форму шара"},
        "Вода закипает при 100 градусах Цельсия": {"truth": True, "explanation": "При нормальном атмосферном давлении"},
        "Солнце вращается вокруг Земли": {"truth": False, "explanation": "Земля вращается вокруг Солнца"},
        "Человек использует только 10% мозга": {"truth": False, "explanation": "Мозг активен полностью, но не все части одновременно"},
        "Изменение климата - это миф": {"truth": False, "explanation": "Научный консенсус подтверждает изменение климата"},
        "Мобильные телефоны вызывают рак мозга": {"truth": "Не доказано", "explanation": "Исследования не выявили четкой связи"}
    }

    similar_facts = []

    # Поиск похожих утверждений
    if sim_model is not None:
        try:
            for demo_fact, info in demo_facts.items():
                embeddings = sim_model.encode([claim, demo_fact], convert_to_tensor=True)
                similarity = util.similarity(embeddings[0], embeddings[1]).item()

                if similarity > 0.4:  # Пониженный порог для лучшего поиска
                    similar_facts.append({
                        "fact": demo_fact,
                        "similarity": round(similarity * 100, 2),
                        "truth": info["truth"],
                        "explanation": info["explanation"]
                    })

            similar_facts.sort(key=lambda x: x["similarity"], reverse=True)
        except Exception as e:
            print(f"Ошибка при поиске похожих фактов: {e}")

    return {
        "claim": claim,
        "category": best_category,
        "confidence": round(confidence * 100, 2),
        "response": response_text,
        "similar_facts": similar_facts[:3]  # Только 3 самых похожих
    }


@app.route('/check_claim', methods=['POST'])
def check_claim():

    data = request.json
    claim = data.get('claim', '').strip()

    if not claim:
        return jsonify({"error": "Пустое утверждение"})

    if len(claim) < 5:
        return jsonify({"error": "Слишком короткое утверждение (минимум 5 символов)"})

    result = analyze_claim_with_ai(claim)
    return jsonify(result)


@app.route('/check_url', methods=['POST'])
def check_url():

    data = request.json
    url = data.get('url', '').strip()

    if not url:
        return jsonify({"error": "Пустой URL"})

    parsed = urllib.parse.urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        return jsonify({"error": "Некорректный URL"})

    article_info = fetch_article_info_simple(url)

    if not article_info['success']:
        return jsonify({"error": f"Не удалось загрузить статью: {article_info.get('error', 'Неизвестная ошибка')}"})

    claims = claims_from_text(article_info['text'])

    checked_claims = []
    for claim in claims[:3]:  # Проверяем первые 3 утверждения
        if len(claim) > 20 and len(claim) < 500:
            try:
                checked_claims.append(analyze_claim_with_ai(claim))
            except Exception as e:
                print(f"Ошибка при проверке утверждения: {e}")
                continue

    result = {
        "article_info": {
            "title": article_info.get("title", ""),
            "url": url
        },
        "total_claims_found": len(claims),
        "checked_claims": checked_claims
    }

    return jsonify(result)


@app.route('/analyze_text', methods=['POST'])
def analyze_text():

    data = request.json
    text = data.get('text').strip()

    if not text or len(text) < 10:
        return jsonify({"error": "Слишком короткий текст (минимум 10 символов)"})

    if len(text) > 10000:
        return jsonify({"error": "Слишком длинный текст (максимум 10000 символов)"})

    # Извлечение утверждений
    claims = claims_from_text(text)

    # Проверка каждого утверждения
    results = []
    for i, claim in enumerate(claims[:5]):
        if 20 < len(claim) < 500:
            try:
                result = analyze_claim_with_ai(claim)
                result['id'] = i + 1
                results.append(result)
            except Exception as e:
                print(f"Ошибка при анализе утверждения: {e}")
                continue

    # Общая оценка достоверности
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)

        # Подсчитываем категории
        category_scores = {
            "правдивый факт": 100,
            "частично правдиво": 60,
            "непроверенное утверждение": 40,
            "вводящее в заблуждение утверждение": 30,
            "ложная информация": 10
        }

        trust_score = sum(category_scores.get(r['category'], 50) for r in results) / len(results)

        # Определяем вердикт
        if trust_score > 70:
            verdict = "Достоверно"
        elif trust_score > 40:
            verdict = "Частично достоверно"
        else:
            verdict = "Требует проверки"
    else:
        avg_confidence = 0
        trust_score = 50
        verdict = "Не удалось проанализировать"

    return jsonify({
        "total_claims": len(results),
        "claims": results,
        "summary": {
            "average_confidence": round(avg_confidence, 2),
            "trust_score": round(trust_score, 2),
            "verdict": verdict
        }
    })


@app.route('/health', methods=['GET'])
def health_check():

    try:
        fact_checker_model, sim_model = initialize_models()
        models_loaded = fact_checker_model is not None and sim_model is not None

        return jsonify({
            "status": "healthy" if models_loaded else "degraded",
            "models_loaded": models_loaded,
            "message": "Факт-чекер готов к работе" if models_loaded else "Некоторые модели не загружены"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "models_loaded": False,
            "error": str(e)
        }), 500


@app.route('/')
@app.route('/index')
def index():

    success = request.args.get('success', False)
    articles = Article.query.order_by(Article.date.desc()).all()
    return render_template("index.html", articles=articles, success=success)


@app.route('/about')
def about():

    return render_template("about.html")


@app.route('/ai', methods=['POST', 'GET'])
def ai():

    if request.method == "POST":
        text = request.form.get('text')

        if not text or text.strip() == "":
            return "Текст не может быть пустым", 400

        article = Article(text=text.strip())

        try:
            db.session.add(article)
            db.session.commit()
            return redirect(url_for('ai', success=True))
        except Exception as e:
            db.session.rollback()
            return render_template("ai.html", error=f"Ошибка сохранения: {str(e)}")
    else:
        success = request.args.get('success', False)
        return render_template("ai.html", success=success)


@app.errorhandler(404)
def not_found_error(error):

    return jsonify({"error": "Ресурс не найден"}), 404


@app.errorhandler(500)
def internal_error(error):

    db.session.rollback()
    return jsonify({"error": "Внутренняя ошибка сервера"}), 500


if __name__ == '__main__':
    import threading


    def preload_models():

        try:
            initialize_models()
            print(" Модели успешно загружены!")
        except Exception as e:
            print(f" Ошибка при загрузке моделей: {e}")


    # Запускаем загрузку моделей в отдельном потоке
    thread = threading.Thread(target=preload_models)
    thread.daemon = True
    thread.start()

    print(" Запуск сервера...")

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
