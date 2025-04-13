import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

def prepare_data(image_metadata_path, images_dir):
    df = pd.read_csv(image_metadata_path)
    df["image_id"] = df["image_id"].astype(str) + ".jpg"  # Adjust image ID format to match filenames
    df = df[df["image_id"].isin(os.listdir(images_dir))]  # Filter dataframe to keep only images that exist in dataset

    le_dx = LabelEncoder()  # Label encoder to convert string labels into numerical labels
    df["label"] = le_dx.fit_transform(df["dx"])  # Convert diagnosis column into numerical labels
    
    train_df, test_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=42)
    
    return df, train_df, test_df, le_dx

def get_transforms(train=True): # Image transformation pipeline
    if train:
        return transforms.Compose([
            transforms.Resize((72, 72)), # Resize for consistency
            transforms.RandomHorizontalFlip(), # Data augmentation for generalization
            transforms.RandomRotation(10), # Slight rotation for robustness
            transforms.CenterCrop((64, 64)), # Crop to standard input size
            transforms.ToTensor(), # Convert image to tensor
            transforms.Normalize([0.5]*3, [0.5]*3), # Normalize to [-1, 1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)), # Direct resize for validation/testing
            transforms.ToTensor(), # Convert image to tensor
            transforms.Normalize([0.5]*3, [0.5]*3), # Normalize to [-1, 1]
        ])

class SkinLesionDataset(Dataset):
    def __init__(self, df, transform, images_dir="data/images"):
        self.df = df.reset_index(drop=True)
        self.transform = transform  # Store transformation function
        self.images_dir = images_dir

    def __len__(self):
        length = len(self.df)  # Number of samples
        return length

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        img_path = os.path.join(self.images_dir, image_id)
        img = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
        label = self.df.iloc[idx]["label"]
        img_transformed = self.transform(img)  # Apply specified transformations to image
        return img_transformed, label

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = self._conv_block(3, 64)
        self.conv2 = self._conv_block(64, 128)
        self.conv3 = self._conv_block(128, 256)

        self.pool = nn.MaxPool2d(2) # Downsample feature maps by factor of 2
        self.dropout1 = nn.Dropout(0.5) # Prevent overfitting
        self.fc1 = nn.Linear(256 * 8 * 8, 512) # First fully connected layer
        self.bn_fc = nn.BatchNorm1d(512) # Normalize layer inputs for faster training
        self.dropout2 = nn.Dropout(0.4) # Further regularization
        self.fc2 = nn.Linear(512, num_classes) # Final output layer

    def _conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # Convolution
            nn.BatchNorm2d(out_channels), # Batch normalization
            nn.ReLU(inplace=True) # ReLU activation
        )
        for m in block: # Apply He initialization to conv layers
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        return block

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x))) # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x))) # Conv3 -> ReLU -> Pool
        x = torch.flatten(x, 1) # Flatten dimensions
        x = self.dropout1(x)
        x = F.relu(self.bn_fc(self.fc1(x))) # FC1 -> BatchNorm -> ReLU
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_cnn_model(train_df, test_df, le_dx, transform, images_dir="data/images", epochs=15, batch_size=32):
    # Define transformations for training and validation datasets
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    # Create datasets and data loaders
    train_ds = SkinLesionDataset(train_df, train_transform, images_dir)
    test_ds = SkinLesionDataset(test_df, test_transform, images_dir)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available

    model = CNN(num_classes=len(le_dx.classes_)).to(device) # Initialize model with correct output size

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.5, 1.5, 3.0, 2.0, 0.5, 2.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) # Adjust learning rate when plateau

    val_losses = []

    for epoch in range(epochs):
        model.train() # Set model to training mode
        train_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(train_dl):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad() # Reset gradients
            outputs = model(imgs) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backpropagate
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent gradient explosion
            optimizer.step() # Update weights
            train_loss += loss.item()

        model.eval() # Set model to evaluation mode
        val_loss = 0.0 # Initialize validation loss
        with torch.no_grad(): # Disable gradient computation for evaluation
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(device), labels.to(device) # Move data to device
                outputs = model(imgs) # Forward pass
                loss = criterion(outputs, labels) # Compute validation loss
                val_loss += loss.item()

        val_losses.append(val_loss)

        scheduler.step(val_loss) # Adjust the learning rate scheduler based on validation loss

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dl):.4f}, Val Loss: {val_loss/len(test_dl):.4f}')

   # Final evaluation on test data
    model.eval() # Ensure model is in eval mode
    preds, targets = [], [] # Initialize lists to store predictions and ground truth
    with torch.no_grad(): # No gradient computation
        for imgs, labels in test_dl:
            imgs = imgs.to(device) # Move images to device
            outputs = model(imgs) # Get model predictions
            pred = outputs.argmax(dim=1).cpu().numpy() # Convert logits to predicted class indices
            preds.extend(pred)
            targets.extend(labels.numpy())

    report = classification_report(targets, preds, target_names=le_dx.classes_, output_dict=True, zero_division=0)
    print('CNN Results:\n', report)
    return report

def train_random_forest_model(train_df, test_df, le_dx, transform, images_dir="data/images"):

    # Initialize empty lists to hold training data and labels
    X_train = []
    y_train = []

    # Loop through each image in the training dataframe
    for i in range(len(train_df)):
        img_path = os.path.join(images_dir, train_df.iloc[i]["image_id"])  # Build image path
        img = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
        img_tensor = transform(img)  # Apply preprocessing transform
        X_train.append(img_tensor.numpy().flatten())  # Flatten image and append to features
        y_train.append(train_df.iloc[i]["label"])  # Append label

    # Repeat for test data
    X_test = []
    y_test = []
    for i in range(len(test_df)):
        img_path = os.path.join(images_dir, test_df.iloc[i]["image_id"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        X_test.append(img_tensor.numpy().flatten())
        y_test.append(test_df.iloc[i]["label"])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=200) # Reduce to 200 principal components
    X_train_pca = pca.fit_transform(X_train) # Fit PCA on training data
    X_test_pca = pca.transform(X_test) # Transform test data using same PCA model

    rf_model = RandomForestClassifier(
        n_estimators=300, # Number of trees
        max_depth=30, # Maximum depth of tree
        min_samples_split=3, # Minimum samples to split an internal node
        n_jobs=-1,
        class_weight='balanced',  # Balance class weights to reduce class imbalance
        random_state=42
    )
    rf_model.fit(X_train_pca, y_train)  # Train the model on PCA-reduced training data

    y_pred = rf_model.predict(X_test_pca) # Make predictions on the test set

    report = classification_report(y_test, y_pred, target_names=le_dx.classes_, output_dict=True, zero_division=0)
    print("Random Forest Results:\n", report)

    return report

#____________________________________________________________________________
# Prepare dataset, split, and label encoder
df, train_df, test_df, le_dx = prepare_data("data/image_metadata.csv", "data/images")

transform = get_transforms()

# Train Random Forest model
print("\nTraining Random Forest model")
rf_report = train_random_forest_model(train_df, test_df, le_dx, transform, "data/images")

# Random Forest metrics
rf_acc = rf_report['accuracy']
rf_recall = rf_report['macro avg']['recall']
rf_f1 = rf_report['macro avg']['f1-score']

# Train CNN model
print("Training CNN model")
cnn_report = train_cnn_model(train_df, test_df, le_dx, transform, "data/images")

# CNN metrics
cnn_acc = cnn_report['accuracy']
cnn_recall = cnn_report['macro avg']['recall']
cnn_f1 = cnn_report['macro avg']['f1-score']

# Comparison of model performance
print(f"\nModel Comparison:")
print(f"CNN Accuracy: {cnn_acc:.4f}")
print(f"CNN Recall (macro): {cnn_recall:.4f}")
print(f"CNN F1 Score (macro): {cnn_f1:.4f}")
print(f"RF Accuracy: {rf_acc:.4f}")
print(f"RF Recall (macro): {rf_recall:.4f}")
print(f"RF F1 Score (macro): {rf_f1:.4f}")

# Show which model outperforms the other
print(f"\nAccuracy Difference: {abs(cnn_acc - rf_acc):.4f} in favor of {('CNN' if cnn_acc > rf_acc else 'RF')}")
print(f"Recall Difference: {abs(cnn_recall - rf_recall):.4f} in favor of {('CNN' if cnn_recall > rf_recall else 'RF')}")
print(f"F1 Score Difference: {abs(cnn_f1 - rf_f1):.4f} in favor of {('CNN' if cnn_f1 > rf_f1 else 'RF')}")