import os

def clean_dir(dir_path): #removes all files from the directory
    if not os.path.isdir(dir_path):
        print(f"Error: '{dir_path}' is not a valid directory.")
        return

    for item_name in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item_name)

        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
            except OSError as e:
                print(f"Error removing file {item_path}: {e}")