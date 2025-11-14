import threading
import os 
from flask import url_for
import time

from crash_summary import create_report

class ReportProcessingThread(threading.Thread): 
    def __init__(self, report_queue, reports_db, db_lock):
        super().__init__()
        self.daemon = True # kills process without letting it finish

        self.report_queue = report_queue
        self.reports_db = reports_db
        self.db_lock = db_lock

    def run(self):
        while True: 
            crash_image_paths = self.report_queue.get()

            try: 
                current_time = time.time()
                crash_description = create_report([crash_image_paths[0]])
                print(f"making the report took {time.time() - current_time}")

                image_filenames = [os.path.basename(path) for path in crash_image_paths]

                new_report = {
                    "description" : crash_description,
                    "images" : image_filenames
                }

                with self.db_lock: 
                    self.reports_db.append(new_report)

            except Exception as e: 
                print(f"report processor thread failed : {e}")

            self.report_queue.task_done()

