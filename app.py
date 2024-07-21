from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, Response, jsonify
from video_monitor import get_abs_path
from db_utils import check_create_database
import sqlite3, os
app = Flask(__name__)
from sqlalchemy import func, case, and_
import logging


db_path = get_abs_path('data', 'brainy_bits.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data/brainy_bits.db'  # Update with your database URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Emotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    emotion = db.Column(db.Float, nullable=False)

class EyeTracking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    direction = db.Column(db.Float, nullable=False)

class HeadPose(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    pose = db.Column(db.Float, nullable=False)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classroom_monitoring')
def classroom_monitoring():
    return render_template('classroom_monitoring.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def get_abs_path(*args):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), *args)

def connect_to_db(db_path):
    db_conn = sqlite3.connect(db_path)
    return db_conn


def fetch_focus_data():
    conn = connect_to_db('data/brainy_bits.db')  # Update with your actual database path
    cursor = conn.cursor()

    # Aggregate the total seconds for each emotion
    cursor.execute("""
        SELECT 
            SUM(Angry_s) AS Angry_s,
            SUM(Sad_s) AS Sad_s,
            SUM(Happy_s) AS Happy_s,
            SUM(Fear_s) AS Fear_s,
            SUM(Disgust_s) AS Disgust_s,
            SUM(Neutral_s) AS Neutral_s,
            SUM(Surprise_s) AS Surprise_s
        FROM emotion_detect_data
    """)
    emotion_totals = cursor.fetchone()

    conn.close()

    if emotion_totals:
        focused = emotion_totals[5]  # Neutral_s
        not_focused = sum(emotion_totals) - focused  # Total - Neutral_s

        return {
            'neutral': focused,
            'not_neutral': not_focused
        }
    else:
        return {
            'neutral': 0,
            'not_neutral': 0
        }

@app.route('/get_data')
def get_data():
    try:
        focus_data = fetch_focus_data()
        return jsonify(focus_data)
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    from video_monitor import generate_frames  # Import inside the function to avoid circular import
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("Starting the app...")
    check_create_database(db_path)
    app.run(debug=True)
