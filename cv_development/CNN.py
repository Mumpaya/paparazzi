import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import cv2

# --- Configuration ---
IMG_DIR = '/home/roan2003/paparazzi/cv_development/cyberzoo_poles/20190121-135009'   # Update this!
CSV_FILE = '/home/roan2003/paparazzi/cv_development/poles_gt.csv'           # Update if you renamed it
BATCH_SIZE = 32                   # Small batch size for small datasets
EPOCHS = 50                      # Number of passes through the data
LEARNING_RATE = 0.001
IMG_SIZE = 128                    # Resize images to 128x128 so the math works

# --- 1. Custom Dataset to load your CSV ---
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get filename from the first column, label from the second
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        
        # Load image (using PIL because torchvision transforms prefer it over cv2)
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. The CNN Class ---1
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Layers (Extracting features like edges, shapes)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Halves the image size -> 64x64
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Halves again -> 32x32
        )
        
        # Fully Connected Layers (Making the final decision)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 32 channels * 32 width * 32 height
            nn.Linear(32 * 32 * 32, 128), 
            nn.ReLU(),
            nn.Dropout(0.5), # Helps prevent overfitting on small datasets!
            nn.Linear(128, num_classes) # Outputs 4 numbers (logits)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. Training Setup & Loop ---
def train_model():
    # Set up device (uses GPU if you have one, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Transforms: Resize, convert to Tensor, and slightly normalize
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load Data
    dataset = CustomImageDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=transform)
    print(f"Loaded {len(dataset)} labeled images.")
    
    # Split into 80% train and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train set: {len(train_dataset)} images | Validation set: {len(val_dataset)} images")

    # Initialize Model, Loss Function, and Adam Optimizer
    model = SimpleCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # The Training Loop
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            # 1. Forward pass (Make predictions)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 2. Backward pass (Calculate gradients)
            optimizer.zero_grad()
            loss.backward()

            # 3. Optimize (Update weights using Adam)
            optimizer.step()

            # Track accuracy and loss
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss_avg = train_loss / len(train_dataloader)
        train_acc = 100 * train_correct / train_total
        
        # Validation Phase (every 2 epochs)
        val_info = ""
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    
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

    print("\nTraining complete!")
    
    # Save the model
    torch.save(model.state_dict(), 'my_cnn_model.pth')
    print("Model saved as 'my_cnn_model.pth'")


# --- 4. Test on Unlabeled Images ---
def test_on_unlabeled(test_dir='/home/roan2003/paparazzi/cv_development/cyberzoo_poles_panels/20190121-140205', 
                      model_path='my_cnn_model.pth', 
                      output_csv='/home/roan2003/paparazzi/cv_development/test_predictions.csv'):
    """
    Test the trained model on unlabeled images from a directory.
    Saves predictions to a CSV file.
    """
    # Check if test directory exists
    if not os.path.isdir(test_dir):
        print(f"Test directory '{test_dir}' not found.")
        return
    
    # Check if model exists
    if not os.path.isfile(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return
    
    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    model = SimpleCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Get all image files from test directory
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(test_files)} test images.")
    
    if len(test_files) == 0:
        print("No images found in test directory.")
        return
    
    # Make predictions
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
                print(f"[{len(results)}/{len(test_files)}] {img_name} -> Prediction: {prediction}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nPredictions saved to '{output_csv}'")
        
        # Show 10 random test images with predictions
        print("\nDisplaying 10 random test images with predictions...")
        sample_results = results_df.sample(n=min(20, len(results_df)), random_state=None)
        
        for idx, (_, row) in enumerate(sample_results.iterrows()):
            img_name = row['filename']
            prediction = int(row['prediction'])
            
            img_path = os.path.join(test_dir, img_name)
            cv_image = cv2.imread(img_path)
            
            if cv_image is None:
                print(f"Could not read {img_name}")
                continue
            
            # Add prediction text to image
            pred_text = f"Prediction: {prediction}"
            cv2.putText(cv_image, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Test Predictions', cv_image)
            
            print(f"[{idx+1}/20] {img_name} -> Prediction: {prediction}")
            print("Press any key to continue, 'q' to quit...")
            
            key = cv2.waitKey(0) & 0xFF
            if chr(key).lower() == 'q':
                break
        
        cv2.destroyAllWindows()
    else:
        print("No predictions to save.")


if __name__ == "__main__":
    print("=== CNN Training and Testing ===")
    print("1. Train model")
    print("2. Test on unlabeled images")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        train_model()
    elif choice == "2":
        test_on_unlabeled()
    else:
        print("Invalid choice.")