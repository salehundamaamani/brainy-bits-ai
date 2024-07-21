from flask import Flask, render_template, Response
from video_monitor import get_abs_path
from db_utils import check_create_database


app = Flask(__name__)

db_path = get_abs_path('data', 'brainy_bits.db')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classroom_monitoring')
def classroom_monitoring():
    return render_template('classroom_monitoring.html')


@app.route('/video_feed')
def video_feed():
    from video_monitor import generate_frames  # Import inside the function to avoid circular import
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data')
def get_data():
    data = []
    with open('data.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'date': row['date'],
                'focused': int(row['focused']),
                'not_focused': int(row['not_focused'])
            })
    return jsonify(data)



if __name__ == '__main__':
    print("Starting the app...")
    check_create_database(db_path)
    app.run(debug=True)
