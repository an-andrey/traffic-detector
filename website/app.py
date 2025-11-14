import os
import queue
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from werkzeug.utils import secure_filename 
import atexit
import time
import threading

#file imports
from video_thread import VideoProcessingThread
from report_thread import ReportProcessingThread
from utils.app_utils import *


# setting paths & config
UPLOAD_FOLDER = 'static/uploads'
VIDEO_UPLOAD_FOLDER = "videos/user_uploads"
VIDEO_DEMO_FOLDER = "videos/demos"

app = Flask(__name__)
app.config['VIDEO_DEMO_FOLDER'] = VIDEO_DEMO_FOLDER
app.config['VIDEO_UPLOAD_FOLDER'] = VIDEO_UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024 # sets file limit to 100mb for videos


# Setting up all utils for threads and their handling
reports_db = [] #stores reports
report_queue = queue.Queue()
db_lock = threading.Lock()

def clean_reports_db(): 
    global reports_db

    with db_lock:
        reports_db.clear() # make sure to not set as = [], as this will over-write the link with report thread

current_video_thread = None # no video initially 

def kill_video_thread(): 
    if current_video_thread:
        current_video_thread.stop()
        current_video_thread.join() # makes sure thread is stopped before closing the main proccess

def start_new_video_thread(video_path): 
    global current_video_thread

    kill_video_thread()
    clean_reports_db()

    current_video_thread = VideoProcessingThread(video_path, report_queue)
    current_video_thread.start()

report_thread = ReportProcessingThread(report_queue, reports_db, db_lock) # constantly keep the report thread, no matter the video
report_thread.start()


atexit.register(kill_video_thread) # when closing main process, shut down the video thread


# app routes
@app.route('/')
def home(): 
    return render_template("home.html")

@app.route('/start_thread', methods=["POST"])
def start_thread(): 
    global current_video_thread
    video_path = None
    
    # Check for demo video first
    if 'demo_video' in request.form:
        filename = request.form['demo_video']
        # Securely join path
        video_path = os.path.join(app.config['VIDEO_DEMO_FOLDER'], filename)
        if not os.path.exists(video_path):
            flash("Demo video not found. Please check configuration.")
            return redirect(url_for('home'))
        else: 
            start_new_video_thread(video_path)
            return redirect(url_for('video'))

    # Check for file upload
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('home'))
        
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], filename)
            file.save(video_path)
            print(f"Starting uploaded video: {video_path}")

            start_new_video_thread(video_path)

            return redirect(url_for("video"))

    if not video_path:
        flash("No video source selected.")
        return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', reports=reversed(reports_db))


def stream_frames(): # grabs the latest frame from the video thread
    while True:
        frame = current_video_thread.get_frame()

        if frame is None: 
            time.sleep(0.1) #give time for thread to process next frame
            continue 

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/video")
def video(): 
    return render_template("video.html")

@app.route("/video_feed")
def video_feed(): 

    return Response(
        stream_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

#Adding global variables to templates
@app.context_processor
def add_global_vars(): 
    is_feed_live = bool(current_video_thread and current_video_thread.is_alive())

    return {
        "is_feed_live": is_feed_live
    }


# --- Run the App ---
if __name__ == '__main__':
    # Ensure the 'static/uploads' directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Run in debug mode for development
    app.run(debug=False, threaded=True)

    # cleaning all previous frames and videos uploaded by the user
    clean_dir(UPLOAD_FOLDER)
    clean_dir(VIDEO_UPLOAD_FOLDER)
