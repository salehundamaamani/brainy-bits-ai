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
    emotion = db.Column(db.String(50), nullable=False)

class EyeTracking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    direction = db.Column(db.String(50), nullable=False)

class HeadPose(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    pose = db.Column(db.String(50), nullable=False)

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
    conn = connect_to_db(db_path)
    cursor = conn.cursor()

    # Fetch data from the database
    cursor.execute("SELECT * FROM eye_track_data")
    eye_data = cursor.fetchall()

    cursor.execute("SELECT * FROM emotion_detect_data")
    emotion_data = cursor.fetchall()

    cursor.execute("SELECT * FROM head_pose_data")
    head_pose_data = cursor.fetchall()

    conn.close()

    focused_count = 0
    not_focused_count = 0

    # Check the criteria for each person
    for i in range(len(eye_data)):
        person_id = eye_data[i][1]  # Assuming Person_ID is at index 1

        # Get corresponding emotion and head pose data for the same person
        emotion = next((x for x in emotion_data if x[1] == person_id), None)
        head_pose = next((x for x in head_pose_data if x[1] == person_id), None)

        if not emotion or not head_pose:
            continue

        # Criteria for focus
        if (emotion[6] == max(emotion[2:9]) and  # Neutral emotion
            head_pose[2] == max(head_pose[2:7]) and  # Facing forward
            eye_data[i][6] == max(eye_data[i][4:7])):  # Looking straight
            focused_count += 1
        else:
            not_focused_count += 1

    return {
        'focused': focused_count,
        'not_focused': not_focused_count
    }

@app.route('/get_data')
def get_data():
    # Define the time period for the analysis (e.g., last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Query to get focused and not focused data for each day
    results = db.session.query(
        func.date(Emotion.timestamp).label('date'),
        func.sum(
            case([(and_(
                    Emotion.Neutral_s > 0.5,
                    EyeTracking.Duration_Looking_Straight > 0.5,
                    HeadPose.Looking_Forward_s > 0.5
                ), 1)],
                else_=0
            )
        ).label('focused'),
        func.sum(
            case([(and_(
                    Emotion.Neutral_s <= 0.5,
                    EyeTracking.Duration_Looking_Straight <= 0.5,
                    HeadPose.Looking_Forward_s <= 0.5
                ), 1)],
                else_=0
            )
        ).label('not_focused')
    ).filter(
        Emotion.timestamp.between(start_date, end_date)
    ).group_by(
        func.date(Emotion.timestamp)
    ).all()

    # Convert the results to a list of dictionaries
    data = []
    for result in results:
        data.append({
            'date': result.date.strftime('%Y-%m-%d'),  # Convert date to string
            'focused': result.focused,
            'not_focused': result.not_focused
        })

    return jsonify(data)


@app.route('/video_feed')
def video_feed():
    from video_monitor import generate_frames  # Import inside the function to avoid circular import
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("Starting the app...")
    check_create_database(db_path)
    app.run(debug=True)
