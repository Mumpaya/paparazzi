import os
import json
import shutil
import pandas as pd
import cv2
from PIL import Image

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# YOLO Imports
from ultralytics import YOLO

# --- Configuration ---
IMG_DIR = '/home/roan2003/paparazzi/cv_development/cyberzoo_poles/20190121-135009'   
CSV_FILE = '/home/roan2003/paparazzi/cv_development/poles_gt.csv'           
YOLO_BASE_DIR = '/home/roan2003/paparazzi/cv_development/yolo_dataset'

BATCH_SIZE = 32                   
EPOCHS = 50                      
LEARNING_RATE = 0.001
IMG_SIZE = 128                    

# ==========================================
# --- 1. PYTORCH CNN CLASSIFICATION ---
# ==========================================

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        # Filter out rows with NaN labels and invalid label values (must be 0, 1, 2, or 3)
        self.img_labels = self.img_labels.dropna(subset=['label'])
        self.img_labels = self.img_labels[self.img_labels['label'].isin([0, 1, 2, 3])]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        label = int(self.img_labels.iloc[idx, 1])
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training CNN on device: {device} ---")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = CustomImageDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=transform)
    print(f"Loaded {len(dataset)} labeled images.")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train set: {len(train_dataset)} images | Validation set: {len(val_dataset)} images")

    model = SimpleCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_dataloader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss_avg = train_loss / len(train_dataloader)
        train_acc = 100 * train_correct / train_total
        
        val_info = ""
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images = images.to(device).float()
                    labels = labels.to(device).long()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss_avg = val_loss / len(val_dataloader)
            val_acc = 100 * val_correct / val_total
            val_info = f" | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%"
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}%{val_info}")

    print("\nCNN Training complete!")
    torch.save(model.state_dict(), 'my_cnn_model.pth')
    print("Model saved as 'my_cnn_model.pth'")


def test_on_unlabeled(test_dir='/home/roan2003/paparazzi/cv_development/cyberzoo_poles_panels/20190121-140205', 
                      model_path='my_cnn_model.pth', 
                      output_csv='/home/roan2003/paparazzi/cv_development/test_predictions.csv'):
    
    if not os.path.isdir(test_dir) or not os.path.isfile(model_path):
        print("Test directory or model file not found. Please train first.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nFound {len(test_files)} test images.")
    
    results = []
    with torch.no_grad():
        for img_name in test_files:
            img_path = os.path.join(test_dir, img_name)
            try:
                pil_image = Image.open(img_path).convert("RGB")
                tensor_image = transform(pil_image).unsqueeze(0).to(device)
                outputs = model(tensor_image)
                prediction = int(torch.argmax(outputs, dim=1).item())
                results.append({'filename': img_name, 'prediction': prediction})
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    
    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Predictions saved to '{output_csv}'")
    else:
        print("No predictions to save.")


# ==========================================
# --- 2. YOLO OBJECT DETECTION PIPELINE ---
# ==========================================

def convert_bbox_to_yolo(bbox_str, img_width, img_height):
    """ Converts {"x1": x, "y1": y, "x2": x, "y2": y} to YOLO [x_center, y_center, width, height] """
    if pd.isna(bbox_str) or str(bbox_str).strip() == "":
        return None
    
    try:
        # Strip outer quotes if they exist, then replace escaped quotes
        clean_str = str(bbox_str).strip().strip("'").strip('"').replace('""', '"')
        
        # Parse the string into a Python dictionary
        bbox = json.loads(clean_str)
        
        # Extract coordinates
        xmin, ymin = bbox["x1"], bbox["y1"]
        xmax, ymax = bbox["x2"], bbox["y2"]
        
        # Calculate YOLO normalized coordinates
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        
    except Exception as e:
        print(f"Error parsing bounding box: {bbox_str} | Error: {e}")
        return None

def prepare_yolo_dataset():
    print("\n--- Preparing YOLO Dataset ---")
    
    images_dir = os.path.join(YOLO_BASE_DIR, 'images/train')
    labels_dir = os.path.join(YOLO_BASE_DIR, 'labels/train')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    df = pd.read_csv(CSV_FILE)
    success_count = 0
    
    for idx, row in df.iterrows():
        img_name = row.iloc[0]
        label = row.iloc[1]
        
        if label == 0: # Skip empty images for YOLO
            continue
            
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            
        shutil.copy(img_path, os.path.join(images_dir, img_name))
        
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)
        
        yolo_lines = []
        
        # Safely fetch columns if they exist in the CSV
        box_1 = row['box_1_coords'] if 'box_1_coords' in row else ""
        box_2 = row['box_2_coords'] if 'box_2_coords' in row else ""
        
        yolo_b1 = convert_bbox_to_yolo(box_1, img_width, img_height) if not pd.isna(box_1) else None
        yolo_b2 = convert_bbox_to_yolo(box_2, img_width, img_height) if not pd.isna(box_2) else None

        # 0 = pole, 1 = gate
        if label == 1 and yolo_b1:
            yolo_lines.append(f"0 {yolo_b1}")
        elif label == 2 and yolo_b1:
            yolo_lines.append(f"1 {yolo_b1}")
        elif label == 3:
            if yolo_b1: yolo_lines.append(f"0 {yolo_b1}")
            if yolo_b2: yolo_lines.append(f"1 {yolo_b2}")

        with open(txt_path, 'w') as f:
            f.write("\n".join(yolo_lines))
            
        success_count += 1
        
    yaml_content = f"""path: {YOLO_BASE_DIR}
train: images/train
val: images/train 

names:
  0: pole
  1: gate
"""
    yaml_path = os.path.join(YOLO_BASE_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"YOLO dataset prepared! Converted {success_count} images to folder: {YOLO_BASE_DIR}")
    return yaml_path

def train_yolo():
    yaml_path = os.path.join(YOLO_BASE_DIR, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        print("YOLO dataset not found. Please run Option 3 first!")
        return
        
    print("\n--- Starting YOLOv8 Training ---")
    model = YOLO('yolov8n.pt') 
    model.train(data=yaml_path, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE)
    print("YOLO Training complete! Check the 'runs/detect/' folder for your weights.")


# ==========================================
# --- 3. MAIN MENU ---
# ==========================================

if __name__ == "__main__":
    while True:
        print("\n=== CNN & YOLO Vision System ===")
        print("1. Train Classification CNN (PyTorch)")
        print("2. Test Classification CNN on unlabeled images")
        print("3. Prepare Data for YOLO (Convert CSV to YOLO format)")
        print("4. Train YOLO Object Detector")
        print("q. Quit")
        
        choice = input("Enter choice: ").strip().lower()
        
        if choice == "1":
            train_model()
        elif choice == "2":
            test_on_unlabeled()
        elif choice == "3":
            prepare_yolo_dataset()
        elif choice == "4":
            train_yolo()
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")