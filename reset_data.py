"""
Reset script - Delete all enrolled faces and trained model
Run this to start fresh
"""

import os
import shutil

def reset_system():
    print("=== Resetting Pill Dispenser System ===\n")
    
    # Files to delete
    files_to_delete = [
        'names.json',
        'trainer.yml',
        'test_face_detection.jpg'
    ]
    
    # Folders to delete
    folders_to_delete = [
        'images',
        'media'
    ]
    
    # Delete files
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"[DELETED] {file}")
        else:
            print(f"[NOT FOUND] {file}")
    
    # Delete folders
    for folder in folders_to_delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[DELETED] {folder}/")
        else:
            print(f"[NOT FOUND] {folder}/")
    
    print("\n=== Reset Complete ===")
    print("All enrolled faces and training data have been deleted.")
    print("You can now start fresh with the GUI.")

if __name__ == "__main__":
    response = input("\nAre you sure you want to delete ALL data? (yes/no): ")
    if response.lower() == 'yes':
        reset_system()
    else:
        print("Reset cancelled.")

