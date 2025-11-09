import os
import shutil
import queue
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from werkzeug.utils import secure_filename 
import atexit
import time
import threading

from video_thread import VideoProcessingThread
from report_thread import ReportProcessingThread

# setting paths
UPLOAD_FOLDER = 'static/uploads'
VIDEO_PATH = "test.mp4"


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Things that are shared between threads
reports_db = [] #stores reports
report_queue = queue.Queue()
db_lock = threading.Lock()


def report_handler(crash_description, image_paths):
    
    web_urls = []

    with app.app_context(): #need context since this will be called by the background task
        for path in image_paths: 
            filename = os.path.basename(path) # extracts the file name from the path
            
            web_url = url_for('static', filename=f"uploads/{filename}")
            web_urls.append(web_url)

        report = {
            "description": crash_description,
            "images": web_urls
        }

        reports_db.append(report)


# Starting the video thread to process the video in the background
video_thread = VideoProcessingThread(VIDEO_PATH, report_queue)
video_thread.start()

report_thread = ReportProcessingThread(report_queue, reports_db, db_lock)
report_thread.start()

def shutdown_thread(): 
    video_thread.stop()
    video_thread.join() # makes sure thread is stopped before closing the main proccess

atexit.register(shutdown_thread) # when closing main process, shut down the thread


@app.route('/')
def dashboard():
    return render_template('dashboard.html', reports=reversed(reports_db))


def stream_frames(): # grabs the latest frame from the video thread
    while True:
        frame = video_thread.get_frame()

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


# --- Run the App ---
if __name__ == '__main__':
    # Ensure the 'static/uploads' directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Run in debug mode for development
    app.run(debug=False, threaded=True)