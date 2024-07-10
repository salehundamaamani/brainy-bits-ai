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


@app.route('/student_list')
def student_list():
    students = [
        {'name': 'John Doe', 'id': '123'},
        {'name': 'Jane Smith', 'id': '456'},
        {'name': 'Emily Davis', 'id': '789'},
    ]
    return render_template('student_list.html', students=students)


if __name__ == '__main__':
    print("Starting the app...")
    check_create_database(db_path)
    app.run(debug=True)
