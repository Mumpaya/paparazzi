import cv2
import os
import pandas as pd
import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

# --- Configuration ---
IMG_DIR = '/home/roan2003/paparazzi/cv_development/cyberzoo_poles/20190121-135009'  # Folder containing your images
OUTPUT_CSV = '/home/roan2003/paparazzi/cv_development/poles_gt.csv'  # Where labels will be saved      
OPTIONS = {'0': 0, '1': 1, '2': 2, '3': 3}
TARGET_COUNT = 100  # Stop after labeling this many new images

def label_images():
    # 1. Get all images
    all_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(all_files)} total images in folder.")
    
    # 2. Check existing progress
    done_files = set()
    initial_count = 0
    overwrite = False
    
    if os.path.isfile(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            # Ensure we only skip if there is actually a label recorded
            if 'filename' in existing_df.columns:
                initial_count = len(existing_df)
                print(f"\nCSV exists with {initial_count} labeled images.")
                choice = input("Do you want to (c)ontinue from previous labels or (o)verwrite? [c/o]: ").strip().lower()
                
                if choice == 'o':
                    overwrite = True
                    print("Will overwrite existing CSV.")
                else:
                    done_files = set(existing_df['filename'].dropna().tolist())
                    print(f"Will continue labeling. Skipping {len(done_files)} already labeled images.")
        except Exception as e:
            print(f"Could not read CSV: {e}")
    
    # Delete CSV if overwriting
    if overwrite and os.path.isfile(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        initial_count = 0

    # 3. Filter and Randomize
    remaining_files = [f for f in all_files if f not in done_files]
    random.shuffle(remaining_files)
    
    if not remaining_files:
        print("Wait! No images left to label. Check if your CSV already has all the filenames.")
        return

    print(f"Starting session... Target: {TARGET_COUNT}")

    count = 0
    cv2.namedWindow('Labeller', cv2.WINDOW_AUTOSIZE)

    for img_name in remaining_files:
        if count >= TARGET_COUNT:
            break

        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_name}")
            continue

        cv2.imshow('Labeller', img)
        
        # Wait for a valid key press (retry on invalid keys)
        quit_requested = False
        while True:
            key = cv2.waitKey(0) & 0xFF
            char_key = chr(key) if key < 128 else ''

            if char_key in OPTIONS:
                label = OPTIONS[char_key]
                
                file_exists = os.path.isfile(OUTPUT_CSV)
                try:
                    with open(OUTPUT_CSV, 'a') as f:
                        if not file_exists:
                            f.write("filename,label\n")
                        f.write(f"{img_name},{label}\n")
                    
                    count += 1
                    print(f"[{count}/{TARGET_COUNT}] Saved: {img_name}")
                except Exception as e:
                    print(f"Error saving {img_name}: {e}")
                break
            
            elif char_key.lower() == 'q':
                print("User quit.")
                quit_requested = True
                break
            
            else:
                print(f"Invalid key '{char_key}'. Press 0-3 to label, or 'q' to quit.")

        if quit_requested:
            break

    cv2.destroyAllWindows()
    
    # Verify by reading CSV
    saved_count = 0
    if os.path.isfile(OUTPUT_CSV):
        try:
            final_df = pd.read_csv(OUTPUT_CSV)
            saved_count = len(final_df)
        except:
            pass
    
    print(f"\n--- Session Summary ---")
    print(f"Session labeled: {count}")
    print(f"Total in CSV now: {saved_count}")
    print(f"Session finished.")


# --- CNN Model Class (same as in CNN.py) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def show_predictions(num_images=5, model_path='my_cnn_model.pth'):
    """
    Load the trained model and display random labeled images with predictions.
    Shows both the model's prediction and the ground truth label.
    """
    # Check if CSV exists
    if not os.path.isfile(OUTPUT_CSV):
        print(f"CSV file '{OUTPUT_CSV}' not found. Please label some images first.")
        return
    
    # Check if model exists
    if not os.path.isfile(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return
    
    # Load data
    try:
        df = pd.read_csv(OUTPUT_CSV)
        if len(df) == 0:
            print("CSV file is empty. Please label some images first.")
            return
        # Filter out rows with missing labels
        df = df.dropna(subset=['label'])
        if len(df) == 0:
            print("No valid labeled images found in CSV. Please label some images first.")
            return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Label mapping
    label_map = {0: "0", 1: "1", 2: "2", 3: "3"}
    
    # Randomly sample images
    sample_df = df.sample(n=min(num_images, len(df)), random_state=None)
    
    print(f"\nShowing {len(sample_df)} random images with predictions...\n")
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_name = row['filename']
        ground_truth = int(row['label'])
        
        img_path = os.path.join(IMG_DIR, img_name)
        
        # Load and prepare image
        try:
            pil_image = Image.open(img_path).convert("RGB")
            tensor_image = transform(pil_image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue
        
        # Get prediction
        with torch.no_grad():
            outputs = model(tensor_image)
            prediction = int(torch.argmax(outputs, dim=1).item())
        
        # Display using OpenCV
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            print(f"Could not read {img_name}")
            continue
        
        # Rotate image to landscape (90 degrees counterclockwise)
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convert to PIL for rotated text
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Prepare text
        gt_text = f"Ground Truth: {label_map[ground_truth]}"
        pred_text = f"Prediction: {label_map[prediction]}"
        color_match = (0, 255, 0) if prediction == ground_truth else (0, 0, 255)  # Green if correct, Red if wrong
        
        # Create a larger font
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            # Try to load a larger system font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            # Fallback to default if font not available
            font = ImageFont.load_default()
        
        # Draw rotated text on PIL image
        # Text is rotated 90 degrees counterclockwise (270 degrees clockwise)
        draw.text((20, 20), gt_text, fill=(255, 255, 255), font=font, angle=270)
        draw.text((20, 100), pred_text, fill=color_match, font=font, angle=270)
        
        # Convert back to OpenCV
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Prediction Comparison', cv_image)
        
        print(f"[{idx+1}/{len(sample_df)}] {img_name}")
        print(f"  Ground Truth: {label_map[ground_truth]} | Prediction: {label_map[prediction]} {'✓' if prediction == ground_truth else '✗'}")
        print(f"  Press 's' to save, 'q' to quit, or any other key to continue...")
        
        key = cv2.waitKey(0) & 0xFF
        key_char = chr(key).lower()
        
        if key_char == 's':
            # Create output directory if it doesn't exist
            output_dir = '/home/roan2003/paparazzi/cv_development/prediction_examples'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save with descriptive filename
            is_correct = "correct" if prediction == ground_truth else "incorrect"
            output_filename = f"{is_correct}_{img_name}"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, cv_image)
            print(f"  ✓ Saved to {output_path}")
        elif key_char == 'q':
            break
    
    cv2.destroyAllWindows()
    print("\nDone showing predictions.")


def show_incorrect_prediction():
    """Find and display an example of a wrong prediction"""
    model_path = '/home/roan2003/paparazzi/my_cnn_model.pth'
    
    # Check if CSV exists
    if not os.path.isfile(OUTPUT_CSV):
        print(f"CSV file '{OUTPUT_CSV}' not found. Please label some images first.")
        return
    
    # Check if model exists
    if not os.path.isfile(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return
    
    # Load data
    try:
        df = pd.read_csv(OUTPUT_CSV)
        df = df.dropna(subset=['label'])
        if len(df) == 0:
            print("No valid labeled images found in CSV.")
            return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Set up device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Label mapping
    label_map = {0: "0", 1: "1", 2: "2", 3: "3"}
    
    print("\nSearching for incorrect predictions...\n")
    
    # Iterate through all images to find one with wrong prediction
    for idx, (_, row) in enumerate(df.iterrows()):
        img_name = row['filename']
        ground_truth = int(row['label'])
        
        img_path = os.path.join(IMG_DIR, img_name)
        
        # Load and prepare image
        try:
            pil_image = Image.open(img_path).convert("RGB")
            tensor_image = transform(pil_image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue
        
        # Get prediction
        with torch.no_grad():
            outputs = model(tensor_image)
            prediction = int(torch.argmax(outputs, dim=1).item())
        
        # Check if prediction is wrong
        if prediction != ground_truth:
            print(f"Found incorrect prediction: {img_name}")
            print(f"  Ground Truth: {label_map[ground_truth]} | Prediction: {label_map[prediction]}")
            
            # Display using OpenCV
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                print(f"Could not read {img_name}")
                continue
            
            # Rotate image to landscape (90 degrees counterclockwise)
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Convert to PIL for rotated text
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Prepare text
            gt_text = f"Ground Truth: {label_map[ground_truth]}"
            pred_text = f"Prediction: {label_map[prediction]}"
            
            # Create a larger font
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            except:
                font = ImageFont.load_default()
            
            # Draw rotated text on PIL image
            draw.text((20, 20), gt_text, fill=(255, 255, 255), font=font, angle=270)
            draw.text((20, 100), pred_text, fill=(0, 0, 255), font=font, angle=270)  # Red for wrong
            
            # Convert back to OpenCV
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            cv2.imshow('Incorrect Prediction Example', cv_image)
            print(f"  Press 's' to save, or any other key to close...")
            
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key).lower()
            
            if key_char == 's':
                # Create output directory if it doesn't exist
                output_dir = '/home/roan2003/paparazzi/cv_development/prediction_examples'
                os.makedirs(output_dir, exist_ok=True)
                
                # Save with descriptive filename
                output_filename = f"incorrect_{img_name}"
                output_path = os.path.join(output_dir, output_filename)
                
                cv2.imwrite(output_path, cv_image)
                print(f"  ✓ Saved to {output_path}")
            
            cv2.destroyAllWindows()
            return
    
    print("No incorrect predictions found! All predictions match ground truth.")


if __name__ == "__main__":
    print("=== Image Labelling & Prediction Tool ===")
    print("1. Label images")
    print("2. Show predictions (requires trained model)")
    print("3. Show incorrect prediction example")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        label_images()
    elif choice == "2":
        num_images = input("How many random images to show? (default 5): ").strip()
        try:
            num_images = int(num_images) if num_images else 5
        except ValueError:
            num_images = 5
        show_predictions(num_images=num_images)
    elif choice == "3":
        show_incorrect_prediction()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")