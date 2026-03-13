"""
Pill Dispenser Face Recognition - Professional GUI
Clean layout with camera feed on the right side
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import json
import os
import threading
import numpy as np
import glob
from PIL import Image, ImageTk

# ========== CONFIGURATION ==========
CAMERA = {'index': 0, 'width': 640, 'height': 480}
FACE_DETECTION = {'scale_factor': 1.1, 'min_neighbors': 6, 'min_size': (50, 50)}
TRAINING = {'samples_needed': 120}
RECOGNITION = {'confidence_threshold': 70, 'high_confidence_threshold': 90}
PATHS = {
    'image_dir': 'images',
    'cascade_file': 'haarcascade_frontalface_default.xml',
    'names_file': 'names.json',
    'trainer_file': 'trainer.yml'
}

# ========== HELPER FUNCTIONS ==========
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_id(directory):
    if not os.path.exists(directory):
        return 1
    user_ids = []
    for filename in os.listdir(directory):
        if filename.startswith('Users-'):
            try:
                user_ids.append(int(filename.split('-')[1]))
            except:
                pass
    return max(user_ids + [0]) + 1

def save_name(face_id, face_name, filename):
    names_json = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read().strip()
            if content:
                names_json = json.loads(content)
    names_json[str(face_id)] = face_name
    with open(filename, 'w') as f:
        json.dump(names_json, f, indent=4, ensure_ascii=False)

def load_names(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}

def get_images_and_labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples, ids = [], []
    detector = cv2.CascadeClassifier(PATHS['cascade_file'])
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split("-")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

def initialize_camera():
    cam = cv2.VideoCapture(CAMERA['index'])
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    return None

# ========== GUI CLASS ==========
class PillDispenserGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pill Dispenser - Face Biometric System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#e8e8e8")
        
        self.is_capturing = False
        self.is_verifying = False
        self.face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        self.current_frame = None
        self.camera = None
        self.frame_count = 0
        
        self.create_widgets()
        self.update_list()
        
        # Start camera
        self.start_camera()
        self.update_camera()
    
    def create_widgets(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="Pill Dispenser - Face Biometric System",
                font=("Arial", 16, "bold"), bg="#2c3e50", fg="white",
                anchor="w").pack(side=tk.LEFT, padx=20, pady=15)
        
        self.status_label = tk.Label(title_frame, text="Ready",
                                     font=("Arial", 12, "bold"), bg="#2c3e50", fg="#2ecc71")
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=15)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#e8e8e8")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Left panel
        left_panel = tk.Frame(main_frame, bg="#d5d5d5", width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 1))
        left_panel.pack_propagate(False)
        
        self.create_left_panel(left_panel)
        
        # Right panel - Camera feed
        right_panel = tk.Frame(main_frame, bg="black")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.create_right_panel(right_panel)
    
    def create_left_panel(self, parent):
        # Face Enrollment Section
        enrollment_frame = tk.LabelFrame(parent, text="Face Enrollment",
                                         font=("Arial", 10, "bold"),
                                         bg="#d5d5d5", fg="#2c3e50",
                                         padx=10, pady=10)
        enrollment_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Name entry
        entry_container = tk.Frame(enrollment_frame, bg="#d5d5d5")
        entry_container.pack(fill=tk.X, pady=5)
        
        tk.Label(entry_container, text="Name:", font=("Arial", 9),
                bg="#d5d5d5").pack(anchor="w")
        
        self.name_entry = tk.Entry(entry_container, font=("Arial", 11),
                                   relief=tk.SUNKEN, bd=2, width=25)
        self.name_entry.pack(fill=tk.X, pady=3)
        
        # Enroll button
        self.enroll_btn = tk.Button(entry_container, text="Enroll Face",
                                    command=self.start_enrollment,
                                    bg="#2ecc71", fg="white",
                                    font=("Arial", 11, "bold"),
                                    relief=tk.RAISED, bd=2,
                                    padx=20, pady=8,
                                    cursor="hand2")
        self.enroll_btn.pack(fill=tk.X, pady=5)
        
        # Verification Section
        verification_frame = tk.LabelFrame(parent, text="Verification",
                                          font=("Arial", 10, "bold"),
                                          bg="#d5d5d5", fg="#2c3e50",
                                          padx=10, pady=10)
        verification_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Verify button
        self.verify_btn = tk.Button(verification_frame, text="Start Verification",
                                   command=self.start_verification,
                                   bg="#3498db", fg="white",
                                   font=("Arial", 11, "bold"),
                                   relief=tk.RAISED, bd=2,
                                   padx=20, pady=8,
                                   cursor="hand2")
        self.verify_btn.pack(fill=tk.X, pady=5)
        
        # Verification status
        self.verify_status = tk.Label(verification_frame, text="Verification: OFF",
                                     font=("Arial", 9),
                                     bg="#d5d5d5", fg="#7f8c8d")
        self.verify_status.pack(pady=5)
        
        # Registered Faces Section
        registered_frame = tk.LabelFrame(parent, text="Registered Faces",
                                        font=("Arial", 10, "bold"),
                                        bg="#d5d5d5", fg="#2c3e50",
                                        padx=10, pady=10)
        registered_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Listbox
        listbox_container = tk.Frame(registered_frame, bg="#d5d5d5")
        listbox_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(listbox_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.face_listbox = tk.Listbox(listbox_container,
                                       font=("Arial", 10),
                                       yscrollcommand=scrollbar.set,
                                       bg="white",
                                       relief=tk.SUNKEN,
                                       bd=2,
                                       selectbackground="#3498db")
        self.face_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.face_listbox.yview)
        
        # Buttons frame
        btn_container = tk.Frame(registered_frame, bg="#d5d5d5")
        btn_container.pack(fill=tk.X, pady=5)
        
        refresh_btn = tk.Button(btn_container, text="Refresh",
                               command=self.update_list,
                               bg="#f39c12", fg="white",
                               font=("Arial", 9, "bold"),
                               relief=tk.RAISED, bd=2,
                               padx=5, pady=6,
                               cursor="hand2")
        refresh_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        train_btn = tk.Button(btn_container, text="Train",
                             command=self.train_model,
                             bg="#9b59b6", fg="white",
                             font=("Arial", 9, "bold"),
                             relief=tk.RAISED, bd=2,
                             padx=5, pady=6,
                             cursor="hand2")
        train_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        delete_btn = tk.Button(btn_container, text="Delete",
                              command=self.delete_user,
                              bg="#e74c3c", fg="white",
                              font=("Arial", 9, "bold"),
                              relief=tk.RAISED, bd=2,
                              padx=5, pady=6,
                              cursor="hand2")
        delete_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Stats
        self.stats_label = tk.Label(registered_frame, text="Users: 0 | Samples: 0",
                                   font=("Arial", 9),
                                   bg="#d5d5d5", fg="#34495e")
        self.stats_label.pack(pady=5)
    
    def create_right_panel(self, parent):
        # Camera label
        self.camera_label = tk.Label(parent, bg="black", text="Loading camera...",
                                     font=("Arial", 14), fg="white")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Instruction text
        instruction = tk.Label(parent, bg="black", fg="white",
                              text="Enter name and click 'Enroll Face' to register a new user",
                              font=("Arial", 10))
        instruction.pack(fill=tk.X, pady=10)
    
    def start_camera(self):
        """Start camera once and keep it open"""
        try:
            self.camera = cv2.VideoCapture(CAMERA['index'])
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        except:
            self.camera = None
    
    def update_camera(self):
        """Update camera feed in GUI - optimized"""
        if self.is_verifying:
            self.root.after(50, self.update_camera)
            return
        
        try:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    # Resize for better performance
                    frame = cv2.resize(frame, (800, 550))
                    
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Only detect faces every 3rd frame for performance
                    self.frame_count += 1
                    if self.frame_count % 3 == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray,
                            scaleFactor=FACE_DETECTION['scale_factor'],
                            minNeighbors=FACE_DETECTION['min_neighbors'],
                            minSize=FACE_DETECTION['min_size'])
                        
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Convert to image and display
                    img = Image.fromarray(frame_rgb)
                    img_tk = ImageTk.PhotoImage(img)
                    self.camera_label.config(image=img_tk)
                    self.camera_label.image = img_tk
                    
                    self.current_frame = frame_rgb
            
            # Schedule next update
            self.root.after(33, self.update_camera)  # ~30 FPS
        except:
            self.camera_label.config(text="Camera not available", image="")
            self.root.after(1000, self.update_camera)
    
    def start_enrollment(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Enter a name")
            return
        
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self.status_label.config(text="Enrolling...", fg="#f39c12")
        self.enroll_btn.config(state="disabled")
        
        threading.Thread(target=self.enroll_face, args=(name,), daemon=True).start()
    
    def enroll_face(self, name):
        try:
            create_directory(PATHS['image_dir'])
            face_id = get_face_id(PATHS['image_dir'])
            save_name(face_id, name, PATHS['names_file'])
            
            cam = initialize_camera()
            if not cam:
                raise ValueError("Camera not available")
            
            count = 0
            
            while self.is_capturing and count < TRAINING['samples_needed']:
                ret, img = cam.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray,
                    scaleFactor=FACE_DETECTION['scale_factor'],
                    minNeighbors=FACE_DETECTION['min_neighbors'],
                    minSize=FACE_DETECTION['min_size'])
                
                for (x, y, w, h) in faces:
                    if count < TRAINING['samples_needed']:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        face_img = gray[y:y+h, x:x+w]
                        cv2.imwrite(f'./{PATHS["image_dir"]}/Users-{face_id}-{count+1}.jpg', face_img)
                        count += 1
                        cv2.putText(img, f"{count}/{TRAINING['samples_needed']}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, name, (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Enrolling', img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            cv2.destroyAllWindows()
            cam.release()
            
            if count >= TRAINING['samples_needed']:
                messagebox.showinfo("Success", f"{name} enrolled! ({count} images)")
            self.update_list()
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.is_capturing = False
            self.status_label.config(text="Ready", fg="#2ecc71")
            self.enroll_btn.config(state="normal")
            self.name_entry.delete(0, tk.END)
    
    def train_model(self):
        if not os.path.exists(PATHS['image_dir']) or len(os.listdir(PATHS['image_dir'])) == 0:
            messagebox.showerror("Error", "No faces enrolled")
            return
        
        try:
            self.status_label.config(text="Training...", fg="#9b59b6")
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            faces, ids = get_images_and_labels(PATHS['image_dir'])
            recognizer.train(faces, np.array(ids))
            recognizer.write(PATHS['trainer_file'])
            num_faces = len(np.unique(ids))
            self.status_label.config(text="Ready", fg="#2ecc71")
            messagebox.showinfo("Success", f"Model trained on {num_faces} faces ✓")
        except Exception as e:
            self.status_label.config(text="Ready", fg="#e74c3c")
            messagebox.showerror("Error", str(e))
    
    def start_verification(self):
        if self.is_verifying:
            self.is_verifying = False
            self.verify_btn.config(text="Start Verification", bg="#3498db")
            self.verify_status.config(text="Verification: OFF", fg="#7f8c8d")
            return
        
        if not os.path.exists(PATHS['trainer_file']):
            messagebox.showerror("Error", "Train the model first")
            return
        
        self.is_verifying = True
        self.verify_btn.config(text="Stop Verification", bg="#e74c3c")
        self.verify_status.config(text="Verification: ON", fg="#2ecc71")
        
        print("\n=== Face Verification Started ===")
        print("Press ESC to stop\n")
        
        threading.Thread(target=self.verify_face, daemon=True).start()
    
    def verify_face(self):
        cam = None
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(PATHS['trainer_file'])
            names = load_names(PATHS['names_file'])
            
            cam = initialize_camera()
            if not cam:
                raise ValueError("Camera not available")
            
            while self.is_verifying:
                ret, img = cam.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray,
                    scaleFactor=FACE_DETECTION['scale_factor'],
                    minNeighbors=FACE_DETECTION['min_neighbors'],
                    minSize=FACE_DETECTION['min_size'])
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    # Enhanced validation for better accuracy
                    # LBPH confidence: lower is better (0-100, where 0 is perfect match)
                    # Check if the predicted ID exists in our registered users
                    is_registered_user = str(id) in names
                    
                    if confidence < RECOGNITION['confidence_threshold'] and is_registered_user:  # Strict threshold + user validation
                        name = names.get(str(id), "Unknown")
                        print(f"[AUTHORIZED] {name} (confidence: {confidence:.1f})")
                        cv2.putText(img, name, (x+5, y-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, f"AUTHORIZED ({confidence:.1f})", (x+5, y+h+25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Additional check: if confidence is very high (>90), it's definitely unknown
                        if confidence > RECOGNITION['high_confidence_threshold'] or not is_registered_user:
                            print(f"[UNAUTHORIZED ACCESS] (confidence: {confidence:.1f}, registered: {is_registered_user})")
                            cv2.putText(img, "UNAUTHORIZED", (x+5, y-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(img, f"Confidence: {confidence:.1f}", (x+5, y+h+25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Medium confidence - show as uncertain
                            print(f"[UNCERTAIN] (confidence: {confidence:.1f})")
                            cv2.putText(img, "UNCERTAIN", (x+5, y-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                            cv2.putText(img, f"Confidence: {confidence:.1f}", (x+5, y+h+25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                cv2.imshow('Verification', img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
        except Exception as e:
            print(f"[ERROR] {str(e)}")
        finally:
            if cam:
                cam.release()
            cv2.destroyAllWindows()
            self.is_verifying = False
            self.verify_btn.config(text="Start Verification", bg="#3498db")
            self.verify_status.config(text="Verification: OFF", fg="#7f8c8d")
            print("\n=== Verification Stopped ===\n")
    
    def update_list(self):
        try:
            names = load_names(PATHS['names_file'])
            count = len(names)
            
            # Get sample count
            sample_count = 0
            if os.path.exists(PATHS['image_dir']):
                sample_count = len([f for f in os.listdir(PATHS['image_dir']) 
                                    if f.endswith('.jpg')])
            
            self.stats_label.config(text=f"Users: {count} | Samples: {sample_count}")
            
            self.face_listbox.delete(0, tk.END)
            for face_id, name in sorted(names.items(), key=lambda x: int(x[0])):
                # Count samples for this user
                user_samples = len([f for f in os.listdir(PATHS['image_dir'])
                                  if f.startswith(f'Users-{face_id}-') and f.endswith('.jpg')])
                self.face_listbox.insert(tk.END, f"{face_id}. {name} ({user_samples} samples)")
        except:
            self.stats_label.config(text="Users: 0 | Samples: 0")
            self.face_listbox.delete(0, tk.END)
    
    def delete_user(self):
        try:
            selection = self.face_listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Select a user to delete")
                return
            
            selected_text = self.face_listbox.get(selection[0])
            face_id = selected_text.split('.')[0]
            
            # Load user name
            names = load_names(PATHS['names_file'])
            user_name = names.get(face_id, "Unknown")
            
            result = messagebox.askyesno("Confirm Delete",
                                        f"Delete user: {user_name}?")
            if not result:
                return
            
            # Delete from names.json
            if face_id in names:
                del names[face_id]
                with open(PATHS['names_file'], 'w') as f:
                    json.dump(names, f, indent=4, ensure_ascii=False)
            
            # Delete images
            image_pattern = f"{PATHS['image_dir']}/Users-{face_id}-*.jpg"
            for image_file in glob.glob(image_pattern):
                os.remove(image_file)
            
            messagebox.showinfo("Success", f"User deleted")
            self.update_list()
            
            # Delete trainer
            if os.path.exists(PATHS['trainer_file']):
                os.remove(PATHS['trainer_file'])
                print("[INFO] Model deleted. Please retrain.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")

# ========== MAIN ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = PillDispenserGUI(root)
    root.mainloop()
