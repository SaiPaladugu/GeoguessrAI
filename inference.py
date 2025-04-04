import os
import random
import requests
import numpy as np
import tensorflow as tf
from io import BytesIO
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.applications.mobilenet_v2 import preprocess_input
from keras._tf_keras.keras.models import Model, load_model
import tkinter as tk
from PIL import Image, ImageTk
from data_collection import generate_random_location, API_KEY, CITY_BOUNDS, is_no_imagery_image

# ─────────────────────────────────────────────────────────────────────────────
# NEW IMPORTS FOR EVALUATION / METRICS
# (Added here so as not to modify existing imports)
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the pre-trained model
model = load_model('best_model.keras')

# Define class mapping (hardcoded based on training data)
class_indices = {'Dubai': 0, 'Ottawa': 1, 'Tokyo': 2}  # Matches training generator
index_to_class = {v: k for k, v in class_indices.items()}

# Function to predict city from an image path
def predict_city(image_path):
    print(f"[DEBUG] Predicting city for image: {image_path}")
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Apply MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_city = index_to_class[predicted_index]
    confidence = prediction[0][predicted_index]
    print(f"[DEBUG] Prediction: {predicted_city} with confidence {confidence:.2f}")
    return predicted_city, confidence

# Function 1: Test on all images in 'testing_images' folder
def test_all_images():
    test_folder = 'testing_images'
    if not os.path.exists(test_folder):
        print(f"Error: '{test_folder}' folder not found.")
        return
    
    results = []
    for filename in os.listdir(test_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_folder, filename)
            predicted_city, confidence = predict_city(image_path)
            result = f"Image: {filename} | Predicted: {predicted_city} | Confidence: {confidence:.2f}"
            print(result)
            results.append((filename, predicted_city, confidence))
    
    if not results:
        print(f"No valid images found in '{test_folder}'.")
    return results

# Custom fetch function for UI to save in root directory
def fetch_random_street_view(city, location, temp_path):
    print(f"[DEBUG] Fetching image for {city} at location {location}")
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        'size': "224x224",  # Match model input size
        'location': f"{location[0]},{location[1]}",
        'heading': 0,
        'pitch': 0,
        'fov': 90,
        'key': API_KEY
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        print(f"[DEBUG] Image fetched successfully, validating...")
        image = Image.open(BytesIO(response.content))
        
        # Validate image *before* saving with adjusted thresholds
        is_invalid = is_no_imagery_image(image, brightness_threshold=220, stddev_threshold=30, unique_colors_threshold=600)
        print(f"[DEBUG] Validation result: {'Invalid' if is_invalid else 'Valid'} image")
        if is_invalid:
            print(f"[⚠] Skipped invalid image for {city}: {location}")
            return False
        
        # Save only if valid
        image.save(temp_path)
        print(f"[✔] Saved temporary image: {temp_path}")
        return True
    else:
        print(f"[✖] Failed to fetch image for {city}: {location}, Status: {response.status_code}")
        return False

# Function 2: UI for random image prediction
class GeoguessrUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Geoguessr AI - City Prediction")
        self.root.geometry("800x600")
        self.root.configure(bg="#A9A9A9")  # Light gray background for cleaner look

        # Center frame for all elements
        self.main_frame = tk.Frame(self.root, bg="#A9A9A9")
        self.main_frame.pack(expand=True)

        # UI elements
        self.image_label = tk.Label(self.main_frame, bg="#A9A9A9")
        self.image_label.pack(pady=20)

        self.prediction_label = tk.Label(self.main_frame, text="Prediction: N/A", font=("Arial", 14), bg="#A9A9A9")
        self.prediction_label.pack(pady=20)

        # Loader label (hidden by default)
        self.loader_label = tk.Label(self.main_frame, text="Loading...", font=("Arial", 12), fg="gray", bg="#A9A9A9")
        
        # Button frame for horizontal alignment
        self.button_frame = tk.Frame(self.main_frame, bg="#A9A9A9")
        self.button_frame.pack(pady=20)

        self.next_button = tk.Button(
            self.button_frame, 
            text="Next Image", 
            command=self.next_image, 
            font=("Arial", 12), 
            bg="#4CAF50",  # Green button
            fg="black", 
            activebackground="#45a049",  # Darker green on click
            padx=10, 
            pady=5
        )
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.quit_button = tk.Button(
            self.button_frame, 
            text="Quit", 
            command=self.quit, 
            font=("Arial", 12), 
            bg="#f44336",  # Red button
            fg="black", 
            activebackground="#da190b",  # Darker red on click
            padx=10, 
            pady=5
        )
        self.quit_button.pack(side=tk.LEFT, padx=10)

        # Initial image load
        self.current_image_path = None
        self.next_image()

        # Bind cleanup to window close
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def fetch_random_image(self):
        # Keep trying until a valid image is fetched
        max_attempts = 10  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            print(f"[DEBUG] Attempt {attempts} to fetch a valid image")
            
            # Randomly select a city and generate an image
            city = random.choice(list(CITY_BOUNDS.keys()))
            location = generate_random_location(city)
            
            # Save to root directory with a unique temp name
            temp_path = f"temp_{city}_{location[0]}_{location[1]}.jpg"
            success = fetch_random_street_view(city, location, temp_path)
            
            if success:
                print(f"[DEBUG] Successfully fetched valid image: {temp_path}")
                return temp_path, city
        
        print(f"[ERROR] Failed to fetch a valid image after {max_attempts} attempts")
        return None, None

    def next_image(self):
        print("[DEBUG] Next Image button pressed")
        # Show loader
        self.loader_label.pack(pady=10)
        self.root.update()  # Force UI update to show loader

        # Remove previous temp image if exists
        if self.current_image_path and os.path.exists(self.current_image_path):
            print(f"[DEBUG] Removing previous image: {self.current_image_path}")
            os.remove(self.current_image_path)

        # Fetch new random image
        image_path, true_city = self.fetch_random_image()
        if not image_path:
            print("[DEBUG] No valid image fetched, updating UI with error message")
            self.prediction_label.config(text="Error: Could not fetch a valid image.")
            self.loader_label.pack_forget()  # Hide loader
            return

        self.current_image_path = image_path
        print(f"[DEBUG] New image path set: {self.current_image_path}")

        # Predict city
        predicted_city, confidence = predict_city(image_path)
        prediction_text = f"Prediction: {predicted_city} (Confidence: {confidence:.2f})\nTrue City: {true_city}"
        self.prediction_label.config(text=prediction_text)
        print(f"[DEBUG] Updated prediction label: {prediction_text}")

        # Display image
        img = Image.open(image_path)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference to avoid garbage collection
        print("[DEBUG] Image displayed in UI")

        # Hide loader
        self.loader_label.pack_forget()
        print("[DEBUG] Loader hidden")

    def quit(self):
        print("[DEBUG] Quit button pressed or window closed")
        # Clean up any remaining temp image before quitting
        if self.current_image_path and os.path.exists(self.current_image_path):
            print(f"[DEBUG] Cleaning up final image: {self.current_image_path}")
            os.remove(self.current_image_path)
        self.root.quit()
        self.root.destroy()

# ─────────────────────────────────────────────────────────────────────────────
# NEW FUNCTION 3: EVALUATE ON 100 TEST IMAGES IN 'testing_images/tests'
# (Generates 6+ metrics and saves all graphs to a new folder in 'scores')
# ─────────────────────────────────────────────────────────────────────────────
def evaluate():
    test_folder = 'testing_images/tests'
    if not os.path.exists(test_folder):
        print(f"Error: '{test_folder}' folder not found.")
        return
    
    # Collect predictions and ground-truth labels
    y_true = []
    y_pred = []
    file_count = 0
    
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_count += 1
            
            # Determine ground truth from filename prefix
            # 25 => Dubai, 35 => Tokyo, 45 => Ottawa
            if filename.startswith("25"):
                true_label = "Dubai"
            elif filename.startswith("35"):
                true_label = "Tokyo"
            elif filename.startswith("45"):
                true_label = "Ottawa"
            else:
                # If the naming doesn't match, skip or mark unknown
                print(f"[WARNING] Filename '{filename}' does not match known prefixes. Skipped.")
                continue
            
            image_path = os.path.join(test_folder, filename)
            pred_label, _ = predict_city(image_path)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
    
    if file_count == 0:
        print(f"No valid images found in '{test_folder}'.")
        return
    
    # Convert to numeric indices for confusion matrix
    y_true_indices = [class_indices[label] for label in y_true]
    y_pred_indices = [class_indices[label] for label in y_pred]

    # Compute metrics
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    acc = accuracy_score(y_true_indices, y_pred_indices)
    prec = precision_score(y_true_indices, y_pred_indices, average=None, zero_division=0)
    rec = recall_score(y_true_indices, y_pred_indices, average=None, zero_division=0)
    f1 = f1_score(y_true_indices, y_pred_indices, average=None, zero_division=0)
    
    # For a textual classification report
    class_report = classification_report(y_true_indices, y_pred_indices, target_names=['Dubai','Ottawa','Tokyo'], zero_division=0)
    
    # Create new folder named by date/time in 'scores'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scores_folder = os.path.join("scores", timestamp)
    os.makedirs(scores_folder, exist_ok=True)
    
    # 1) Save confusion matrix plot
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_indices))
    plt.xticks(tick_marks, ['Dubai','Ottawa','Tokyo'], rotation=45)
    plt.yticks(tick_marks, ['Dubai','Ottawa','Tokyo'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Save plot
    cm_path = os.path.join(scores_folder, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_path}")
    
    # 2) Save classification report as txt
    report_path = os.path.join(scores_folder, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write(class_report)
    print(f"[INFO] Classification report saved to {report_path}")
    
    # 3) Accuracy plot (single bar)
    plt.figure()
    plt.bar(["Accuracy"], [acc])
    plt.title("Overall Accuracy")
    acc_path = os.path.join(scores_folder, "accuracy.png")
    plt.savefig(acc_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Accuracy bar chart saved to {acc_path}")
    
    # 4) Precision by class
    plt.figure()
    plt.bar(['Dubai','Ottawa','Tokyo'], prec)
    plt.title("Precision by Class")
    prec_path = os.path.join(scores_folder, "precision_by_class.png")
    plt.savefig(prec_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Precision by class chart saved to {prec_path}")
    
    # 5) Recall by class
    plt.figure()
    plt.bar(['Dubai','Ottawa','Tokyo'], rec)
    plt.title("Recall by Class")
    rec_path = os.path.join(scores_folder, "recall_by_class.png")
    plt.savefig(rec_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Recall by class chart saved to {rec_path}")
    
    # 6) F1-score by class
    plt.figure()
    plt.bar(['Dubai','Ottawa','Tokyo'], f1)
    plt.title("F1-Score by Class")
    f1_path = os.path.join(scores_folder, "f1_by_class.png")
    plt.savefig(f1_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] F1-score by class chart saved to {f1_path}")
    
    print("\nEvaluation complete!")
    print(f"Results have been saved to: {scores_folder}")

# Main execution
if __name__ == "__main__":
    print("Choose an option:")
    print("1: Test all images in 'testing_images' folder")
    print("2: Open UI for random image prediction")
    print("3: Evaluate model (generates metrics and charts)")
    
    choice = input("Enter 1, 2, or 3: ").strip()
    
    if choice == "1":
        test_all_images()
    elif choice == "2":
        root = tk.Tk()
        app = GeoguessrUI(root)
        root.mainloop()
    elif choice == "3":
        evaluate()
    else:
        print("Invalid choice. Please run again and select 1, 2 or 3.")
