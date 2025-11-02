import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'website/static/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# This will still be our simple in-memory "database"
reports_db = []


@app.route('/')
def dashboard():
    return render_template('dashboard.html', reports=reversed(reports_db))

@app.route('/api/add_report', methods=['POST'])
def api_add_report():
    # Get the JSON data sent from the client
    data = request.get_json()
    
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400
        
    description = data.get('description')
    original_image_paths = data.get('image_paths', []) # Default to empty list
    
    web_image_urls = []

    
    for original_path in original_image_paths:
        if not os.path.exists(original_path):
            print(f"Warning: File not found at path {original_path}. Skipping.")
            continue
            
        try:
            # Get the original filename (e.g., "crash_scene_1.jpg")
            filename = os.path.basename(original_path)
            # Make sure the filename is safe for the web
            safe_filename = secure_filename(filename)
            
            # Create the full destination path
            # e.g., 'static/uploads/crash_scene_1.jpg'
            destination_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            
            # Copy the file
            shutil.copy(original_path, destination_path)
            
            # Create the web-accessible URL
            # e.g., '/static/uploads/crash_scene_1.jpg'
            web_url = url_for('static', filename=f'uploads/{safe_filename}')
            web_image_urls.append(web_url)
            
        except Exception as e:
            print(f"Error copying file {original_path}: {e}")

    # 3. Create the new report object (as a dictionary)
    new_report = {
        'description': description,
        'images': web_image_urls  # The list of *new* web URLs
    }
    
    # 4. "Save" the report to our in-memory list
    reports_db.append(new_report)
    
    # 5. Send a success response back to the client
    return jsonify({
        "status": "success",
        "message": "Report added",
        "report_id": len(reports_db) # Just as an example
    }), 201


# --- Run the App ---
if __name__ == '__main__':
    # Ensure the 'static/uploads' directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Run in debug mode for development
    app.run(debug=True)