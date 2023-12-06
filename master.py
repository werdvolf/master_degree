
import numpy as np
from PIL import Image
import os
from pathlib import Path
from math import isnan
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from collections import Counter
import random


import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import scipy



LR = 0.00001
THRESHHOLD = 0.5
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
BALANCED = True
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

## LOADING DATA

datasets_path = './data/'
datasets_name = ["G1020", "ORIGA", "REFUGE"]

image_folder = '/Images_Square/'
mask_folder = '/Masks_Square/'

csv_path = 'master.csv'

train_dirs = [datasets_path + x for x in datasets_name]


class GlaucomaDataset(Dataset):
    def __init__(self, root_dirs, image_transform=None, mask_transform=None, mode='train', task='segmentation', data_transform_augmentation=None):
        self._classes = {'No Glaucoma': 0, 'Glaucoma': 0}
        self.mode = mode
        self.task = task
        self.root_dirs = root_dirs
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.data_transform_augmentation = data_transform_augmentation
        if task == 'segmentation':
            self.images, self.od, self.oc, self.segs, self.labels = self.load_data()
        else:
            self.images, self.labels = self.load_data()    
    def load_data(self):
        df = pd.read_csv(csv_path)
        if self.task == 'segmentation':
            images, od, oc, segs, labels = [], [], [], [], []
            for root_dir in self.root_dirs:
                image_files = sorted(os.listdir(root_dir + image_folder))
                mask_files = sorted(os.listdir(root_dir + mask_folder))
                for image_file, mask_file in zip(image_files, mask_files):
                    q = root_dir[6:] + '/' + image_file
                    filtered_rows = df[df['ImgName'] == q]
                    label = filtered_rows['Label'].values[0]
                    if self.mode == 'train':
                        if not isnan(label):
                            label = int(label)
                            labels.append(label)
                            if label == 1:
                                self._classes['Glaucoma'] += 1
                            else: 
                                self._classes['No Glaucoma'] += 1
                        else:
                            continue
                    elif self.mode == 'test':
                        if isnan(label):
                            label = None
                            labels.append(label)
                            continue
                        else:
                            continue

                    image = Image.open(root_dir + image_folder + image_file).convert("RGB")
                    mask = np.array(Image.open(root_dir + mask_folder + mask_file).convert("L"))
                    od_image = Image.fromarray((mask == 1).astype(np.float32))
                    oc_image = Image.fromarray((mask == 2).astype(np.float32))

                    if self.image_transform is not None:
                        image = self.image_transform(image)

                    if self.mask_transform is not None:
                        od_image = self.mask_transform(od_image)
                        oc_image = self.mask_transform(oc_image)

                    seg = torch.cat([od_image, oc_image], dim=0)
                    images.append(image)
                    od.append(od_image)
                    oc.append(oc_image)
                    segs.append(seg)

            return images, od, oc, segs, labels
        

    def balance_data(self, X, y, data_transforms, augmentation=None):
        classes_distribution = Counter(y)
        minority_class = min(classes_distribution.values())
        majority_class = max(classes_distribution.values())

        # Determine the desired number of samples for the minority class
        desired_minority_samples = majority_class - minority_class  # Balancing with the majority class

        # Create a list of indices for the minority class ('Glaucoma')
        minority_class_indices = [i for i, label in enumerate(y) if label == 1]

        # Oversample the minority class while applying data augmentation
        oversampled_minority_indices = random.choices(minority_class_indices, k=desired_minority_samples)


        for index in oversampled_minority_indices:
            original_image = X[index]
            augmented_image = data_transforms(original_image)  # Apply data augmentation to the image
            label = y[index]
            X.append(augmented_image)
            y.append(label)

        if augmentation is not None:
            augmentation_indices = [i for i in range(len(y))]
            augmentation_indeces_to_transform = random.choices(augmentation_indices, k=augmentation)
            for index in augmentation_indeces_to_transform:
                original_image = X[index]
                augmented_image = data_transforms(original_image)  # Apply data augmentation to the image
                label = y[index]
                X.append(augmented_image)
                y.append(label)
        # Extend the original data with the augmented data

        data = list(zip(X, y))

        # Shuffle the combined data
        random.shuffle(data)

        # Split the shuffled data back into X_balanced and y_balanced
        X, y = zip(*data)

        # Convert the shuffled data back to lists or arrays if needed
        return X, y
    
    def one_hot_encode_labels(self, label):
        one_hot_labels = torch.zeros(1, 2)  # Assuming two classes (0 and 1)
        one_hot_labels[range(1), int(label)] = 1
        return one_hot_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.task == 'classification':
            segs = self.segs[idx]
            label = self.labels[idx]
            label = self.one_hot_encode_labels(label)
            return segs, label
        elif self.task == 'segmentation':
            image = self.images[idx]
            od = self.od[idx]
            oc = self.oc[idx]
            segs = self.segs[idx]
            label = self.labels[idx]
            label = self.one_hot_encode_labels(label)
            return image, od, oc, segs, label



image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images and masks to a consistent size
    transforms.ToTensor(),           # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=mean, std=std)  # Normalize images
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor()
])

data_transform_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=(-30, 30)),  # Rotate images by up to 40 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.8, 1.2)),  # Random crop and resize
    transforms.RandomVerticalFlip(),  # Randomly flip vertically
])

# Define the denormalization transform
denormalize = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1 / s for s in std]),  # Denormalize
    transforms.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),    
])



dataset = GlaucomaDataset(train_dirs, image_transform, mask_transform, mode='train', task='segmentation')

print(f"Train segmentation dataset length - {len(dataset)}")



def plot_image_with_masks(image=None, od_mask=None, oc_mask=None):

    num_plots = sum(x is not None for x in [image, od_mask, oc_mask])

    if num_plots == 0:
        raise ValueError("At least one of 'image', 'od_mask', or 'oc_mask' must be provided.")

    _, axs = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))

    plot_idx = 0

    if image is not None:
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(denormalize(image))
        axs[plot_idx].imshow(image)
        axs[plot_idx].set_title('Image')
        axs[plot_idx].axis('off')
        plot_idx += 1

    if od_mask is not None:
        if isinstance(od_mask, torch.Tensor):
            od_mask = transforms.ToPILImage()(od_mask)
        axs[plot_idx].imshow(od_mask, cmap='gray')
        axs[plot_idx].set_title('Optic Disc Mask')
        axs[plot_idx].axis('off')
        plot_idx += 1

    if oc_mask is not None:
        if isinstance(oc_mask, torch.Tensor):
            oc_mask = transforms.ToPILImage()(oc_mask)
        axs[plot_idx].imshow(oc_mask, cmap='gray')
        axs[plot_idx].set_title('Optic Cup Mask')
        axs[plot_idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_classes(dataset):
    # Assuming 'dataset' has a 'labels' attribute containing the class labels
    labels = dataset.labels

    # Convert class labels to numerical values
    class_labels = [int(label) for label in labels]

    # Calculate the class distribution
    class_counts = [class_labels.count(0), class_labels.count(1)]
    
    # Define class names for the plot
    class_names = ['No Glaucoma', 'Glaucoma']

    # Plot the class distribution
    fig, ax = plt.subplots()
    # plt.figure(figsize=(10, 10))
    ax.bar(class_names, class_counts, color=['blue', 'orange'])
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in the Dataset')

    # Display counts as legends
    for i, count in enumerate(class_counts):
        plt.text(i, count + 50, str(count), ha='center', va='bottom')

    plt.ylim(0, 2100)
    plt.show()



plot_image_with_masks(dataset.images[0], dataset.od[0], dataset.oc[0])
plot_classes(dataset)


## SEGMENTATION MODEL
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epoch = 0

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)
        factor = 2 
        self.down6 = Down(2048, 4096 // factor)
        self.up1 = Up(4096, 2048 // factor)
        self.up2 = Up(2048, 1024 // factor)
        self.up3 = Up(1024, 512 // factor)
        self.up4 = Up(512, 256 // factor)
        self.up5 = Up(256, 128 // factor)
        self.up6 = Up(128, 64)
        self.output_layer = OutConv(64, n_classes)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        out = self.up1(x7, x6)
        out = self.up2(out, x5)
        out = self.up3(out, x4)
        out = self.up4(out, x3)
        out = self.up5(out, x2)
        out = self.up6(out, x1)
        out = self.output_layer(out)
        out = torch.sigmoid(out)
        return out

    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            
        )
        

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        


class OutConv(nn.Module):
    '''
    Simple convolution.
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        

    def forward(self, x):
        return self.conv(x)

def compute_dice_coef(input, target):
    '''
    Compute dice score metric.
    '''
    batch_size = input.shape[0]
    return sum([dice_coef_sample(input[k,:,:], target[k,:,:]) for k in range(batch_size)])/batch_size

def dice_coef_sample(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return (2. * intersection) / (iflat.sum() + tflat.sum())

def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    '''

    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=1)
    # pick the maximum value
    diameter = np.max(vertical_axis_diameter, axis=1)
    # return it
    return diameter

def vertical_cup_to_disc_ratio(od, oc):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    '''
    # compute the cup diameter
    cup_diameter = vertical_diameter(oc)
    # compute the disc diameter
    disc_diameter = vertical_diameter(od)
    return cup_diameter + EPS / (disc_diameter + EPS)

def compute_vCDR_error(pred_od, pred_oc, gt_od, gt_oc):
    '''
    Compute vCDR prediction error, along with predicted vCDR and ground truth vCDR.
    '''
    pred_vCDR = vertical_cup_to_disc_ratio(pred_od, pred_oc)
    gt_vCDR = vertical_cup_to_disc_ratio(gt_od, gt_oc)
    vCDR_err = np.mean(np.abs(pred_vCDR - gt_vCDR))
    return vCDR_err, pred_vCDR, gt_vCDR

# Only retain the biggest connected component of a segmentation map.
def refine_seg(pred):
    np_pred = pred.numpy()
        
    largest_ccs = []
    for i in range(np_pred.shape[0]):
        labeled, ncomponents = scipy.ndimage.label(np_pred[i,:,:])
        bincounts = np.bincount(labeled.flat)[1:]
        if len(bincounts) == 0:
            largest_cc = labeled == 0
        else:
            largest_cc = labeled == np.argmax(bincounts)+1
        largest_cc = torch.tensor(largest_cc, dtype=torch.float32)
        largest_ccs.append(largest_cc)
    largest_ccs = torch.stack(largest_ccs)
    
    return largest_ccs

in_channels = 3  # Assuming RGB images
out_channels = 2  # Number of output channels for od and oc
# Define the model

segmentation_model = UNet(in_channels, out_channels).to(DEVICE)

# Define loss function and optimizer for 2 channels
segmantation_criterion = nn.BCELoss(reduction='mean').to(DEVICE)  # Binary Cross-Entropy loss
segmantation_optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=LR)


# Split the dataset into train, validation, and test sets
total_data = len(dataset)
train_size = int(0.7 * total_data)  # 70% for training
val_size = int(0.15 * total_data)   # 15% for validation
test_size = total_data - train_size - val_size  # The rest for testing
# Use random_split to perform the split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader instances for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Print the sizes of each split
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

def train_segmentation_epoch(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0.
    train_dsc_od = 0.
    train_dsc_oc = 0.
    train_vCDR_error = 0.
    nb_train_batches = len(train_loader)

    for _, (image, od, oc, segs, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        segs = segs.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, segs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / nb_train_batches

        with torch.no_grad():
            pred_od = refine_seg((outputs[:, 0, :, :] >= THRESHHOLD).type(torch.int8).cpu()).to(DEVICE)
            pred_oc = refine_seg((outputs[:, 1, :, :] >= THRESHHOLD).type(torch.int8).cpu()).to(DEVICE)
            od = np.squeeze(od, axis=1).type(torch.int8).to(DEVICE)
            oc = np.squeeze(oc, axis=1).type(torch.int8).to(DEVICE)

            dice_score_train_od = compute_dice_coef(pred_od, od).item() / nb_train_batches
            dice_score_train_oc = compute_dice_coef(pred_oc, oc).item() / nb_train_batches

            vCDR_error, _, _ = compute_vCDR_error(pred_od.cpu().numpy(), pred_oc.cpu().numpy(), od.cpu().numpy(), oc.cpu().numpy())
            train_dsc_od += dice_score_train_od
            train_dsc_oc += dice_score_train_oc
            train_vCDR_error += vCDR_error / nb_train_batches

    return train_loss, train_dsc_od, train_dsc_oc, train_vCDR_error



def validate_segmentation_epoch(model, val_loader, criterion, best_val_auc):
    model.eval()
    val_loss = 0.
    val_dsc_od = 0.
    val_dsc_oc = 0.
    val_vCDR_error = 0.
    nb_val_batches = len(val_loader)

    with torch.no_grad():
        val_data = iter(val_loader)
        for _ in range(nb_val_batches):
            image, od, oc, segs, label = next(val_data)
            image = image.to(DEVICE)
            segs = segs.to(DEVICE)
            outputs = model(image)
            loss = criterion(outputs, segs)
            val_loss += loss.item() / nb_val_batches

            pred_od = refine_seg((outputs[:, 0, :, :] >= THRESHHOLD).type(torch.int8).cpu()).to(DEVICE)
            pred_oc = refine_seg((outputs[:, 1, :, :] >= THRESHHOLD).type(torch.int8).cpu()).to(DEVICE)
            od = np.squeeze(od, axis=1).type(torch.int8).to(DEVICE)
            oc = np.squeeze(oc, axis=1).type(torch.int8).to(DEVICE)

            dice_score_val_od = compute_dice_coef(pred_od, od).item() / nb_val_batches
            dice_score_val_oc = compute_dice_coef(pred_oc, od).item() / nb_val_batches

            vCDR_error, _, _ = compute_vCDR_error(pred_od.cpu().numpy(), pred_oc.cpu().numpy(), od.cpu().numpy(), oc.cpu().numpy())
            val_dsc_od += dice_score_val_od
            val_dsc_oc += dice_score_val_oc
            val_vCDR_error += vCDR_error / nb_val_batches

            
    if val_dsc_od + val_dsc_oc > best_val_auc:
        torch.save(model.state_dict(), 'unet_best_model.pth')
        best_val_auc = val_dsc_od + val_dsc_oc
        print('Best validation AUC reached. Saved model weights.')

    return val_loss, val_dsc_od, val_dsc_oc, val_vCDR_error

# train and validate segmentation model
num_epochs = 40
best_val_auc = 0.
# Create dictionaries to store results
losses = {"train": [], "val": []}
dice_scores_od = {"train": [], "val": []}
dice_scores_oc = {"train": [], "val": []}
vCDR_errors = {"train": [], "val": []}
train_classification_losses = {"train": [], "val": []}

# Training loop
for epoch in range(num_epochs):
    # Training
    train_loss, train_dsc_od, train_dsc_oc, train_vCDR_error, train_classification_loss  = train_segmentation_epoch(segmentation_model, train_loader, segmantation_optimizer, segmantation_criterion, device)
    losses["train"].append(train_loss)
    dice_scores_od["train"].append(train_dsc_od)
    dice_scores_oc["train"].append(train_dsc_oc)
    vCDR_errors["train"].append(train_vCDR_error)

    # Validation
    val_loss, val_dsc_od, val_dsc_oc, val_vCDR_error = validate_segmentation_epoch(segmentation_model, val_loader, segmantation_criterion, device, best_val_auc)
    losses["val"].append(val_loss)
    dice_scores_od["val"].append(val_dsc_od)
    dice_scores_oc["val"].append(val_dsc_oc)
    vCDR_errors["val"].append(val_vCDR_error)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}")
    print(f"Train Dice Score OD: {train_dsc_od:.4f}\tVal Dice Score OD: {val_dsc_od:.4f}")
    print(f"Train Dice Score OC: {train_dsc_oc:.4f}\tVal Dice Score OC: {val_dsc_oc:.4f}")
    print(f"Train vCDR error: {train_vCDR_error:.4f}\tVal vCDR error: {val_vCDR_error:.4f}")
    print("------------------------------------")



segmentation_model.load_state_dict(torch.load('unet_best_model_1.pth'))

# ## CLASSIFICATION MODEL

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)  # Additional convolutional layer
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)  # Additional convolutional layer
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 256)  # Increase the number of neurons
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    

if BALANCED:
    print(f"Size before balancing - {len(dataset)} Class distribution {Counter(dataset.labels)}")
    dataset.task = 'classification'
    dataset.data_transform_augmentation = data_transform_augmentation
    dataset.segs, dataset.labels = dataset.balance_data(dataset.segs, dataset.labels, data_transforms=data_transform_augmentation, augmentation=2000)
    print(f"Size after balancing - {len(dataset)} Class distribution {Counter(dataset.labels)}")
# Split the dataset into train and validation sets using random_split

train_ratio = 0.8  # 70% for training
val_ratio = 0.2  # 15% for validation

# Calculate the number of samples for each split
train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
# Create random splits
train__dataset, val_dataset = random_split(dataset, [train_size, val_size+1])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Print the sizes of each split
print(f"Train dataset size: {train_size}")
print(f"Validation dataset size: {val_size}")




def train_classification_with_segmentation(train_loader, classification_model, segmentation_model, optimizer, criterion, scheduler):
    classification_model.train()
    train_loss = 0
    nb_train_batches = len(train_loader)
    all_predicted = []
    all_labels = []

    with torch.set_grad_enabled(True):
        for segs, label in train_loader:

            label = label.view(-1, 2).to(DEVICE)
            # Perform segmentation using the segmentation model
            with torch.no_grad():
                segmentation_outputs = segmentation_model(image)
                pred_od = refine_seg((segmentation_outputs[:, 0, :, :] >= THRESHHOLD).type(torch.int8).cpu()).to(DEVICE)
                pred_oc = refine_seg((segmentation_outputs[:, 1, :, :] >= THRESHHOLD).type(torch.int8).cpu()).to(DEVICE)
                segmentation_outputs = torch.cat((pred_od.unsqueeze(1), pred_oc.unsqueeze(1)), dim=1)
                
            # Prepare the input for the classification model
            classification_inputs = segmentation_outputs

            # Forward pass through the classification model
            classification_outputs = classification_model(classification_inputs)
            # Compute the classification loss
            loss = criterion(classification_outputs, label)
            
            optimizer.zero_grad()
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update the training loss
            train_loss += loss.item() / nb_train_batches

            # Calculate accuracy
            _, binary_predictions = torch.max(classification_outputs, 1)
            _, true_labels = torch.max(label, 1)
            all_predicted.extend(binary_predictions.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    # Calculate accuracy as a ratio of correct predictions to the total number of predictions
    train_accuracy = accuracy_score(all_labels, all_predicted)
    # Calculate F1 score, precision, recall
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predicted, average='weighted')

    return train_loss, train_accuracy, f1, precision, recall


def validate_classification_with_segmentation(val_loader, classification_model, segmentation_model, criterion, best_val_auc , threshold=0.):
    classification_model.eval()
    val_loss = 0
    nb_val_batches = len(val_loader)
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for  segs, label in val_loader:
            # image = image.to(DEVICE)
            label = label.view(-1, 2).to(DEVICE)

            # Perform segmentation using the segmentation model
            with torch.no_grad():
                segmentation_outputs = segmentation_model(image)
                pred_od = refine_seg((segmentation_outputs[:, 0, :, :] >= THRESHHOLD).type(torch.int8).cpu()).to(DEVICE)
                pred_oc = refine_seg((segmentation_outputs[:, 1, :, :] >= THRESHHOLD).type(torch.int8).cpu()).to(DEVICE)
                segmentation_outputs = torch.cat((pred_od.unsqueeze(1), pred_oc.unsqueeze(1)), dim=1)

            # Prepare the input for the classification model
            classification_inputs = segmentation_outputs

            # Forward pass through the classification model
            classification_outputs = classification_model(classification_inputs)

            # Compute the validation loss
            loss = criterion(classification_outputs, label)
            
            # Update the validation loss
            val_loss += loss.item() / nb_val_batches

        
            # Calculate accuracy, f1, precision, and recall
            _, binary_predictions = torch.max(classification_outputs, 1)  # Assuming a threshold of 0 for binary classification
            _, true_labels = torch.max(label, 1)

            all_predicted.extend(binary_predictions.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            

    # Calculate accuracy as a ratio of correct predictions to the total number of predictions
    val_accuracy = accuracy_score(all_labels, all_predicted)
    # Calculate F1 score, precision, recall
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predicted, average='weighted')

    if val_accuracy > best_val_auc:
        torch.save(classification_model.state_dict(), f'{classification_model.__class__.__name__}.pth')
        best_val_auc = val_accuracy
        print('Best validation accuracy reached. Saved model weights.')

    return val_loss, val_accuracy, f1, precision, recall, best_val_auc

def plot_metrics(metrics_dict, metric_names, plot_title):
    # Create subplots for each metric
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(1, num_metrics, figsize=(16, 4))

    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        for split, values in metrics_dict[metric_name].items():
            ax.plot(range(1, len(values) + 1), values, label=split)

        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid()

    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.show()


from torchvision import  models

classification_models = {
    'custom model': ClassificationModel(),
    'resnet18': models.resnet18(weights='ResNet18_Weights.DEFAULT'),
    'resnet50': models.resnet50(weights='ResNet50_Weights.DEFAULT'),
    'alexnet': models.alexnet(weights='AlexNet_Weights.DEFAULT'),
    'vgg16': models.vgg16(weights='VGG16_Weights.DEFAULT'),
    }


def modify_model(model_name, model):
    num_input_channels = 2
    num_output_channels = 2

    if model_name != 'custom model':
        for param in model.parameters():
            param.requires_grad = False
    if model_name == 'vgg16':
        model.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_output_channels),
                                             nn.Sigmoid()
                                             )
    elif model_name == 'alexnet':
        model.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=11, stride=4, padding=2)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_output_channels),
                                             nn.Sigmoid()
                                             )
    elif model_name in ['resnet18', 'resnet50']:
        model.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_output_channels),
                                  nn.Sigmoid()
                                  )

    model = model.to(DEVICE)
    # criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    criterion = nn.BCELoss().to(DEVICE)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, scheduler


def train_and_validate_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler):
    classification_metrics_dict = {
        "Loss": {"train": [], "val": []},
        "Accuracy": {"train": [], "val": []},
        "F1 Score": {"train": [], "val": []},
        "Precision": {"train": [], "val": []},
        "Recall": {"train": [], "val": []}
    }
    best_val_auc = 0
    best_val_counter = 0
    patience = 5

    for epoch in range(num_epochs):

        train_loss, train_accuracy, train_f1, train_precision, train_recall = train_classification_with_segmentation(train_loader, model, segmentation_model, optimizer, criterion, scheduler)
        classification_metrics_dict["Loss"]["train"].append(train_loss)
        classification_metrics_dict["Accuracy"]["train"].append(train_accuracy)
        classification_metrics_dict["F1 Score"]["train"].append(train_f1)
        classification_metrics_dict["Precision"]["train"].append(train_precision)
        classification_metrics_dict["Recall"]["train"].append(train_recall)


        val_loss, val_accuracy, val_f1, val_precision, val_recall, best_val_auc = validate_classification_with_segmentation(val_loader, model, segmentation_model, criterion, best_val_auc)
        classification_metrics_dict["Loss"]["val"].append(val_loss)
        classification_metrics_dict["Accuracy"]["val"].append(val_accuracy)
        classification_metrics_dict["F1 Score"]["val"].append(val_f1)
        classification_metrics_dict["Precision"]["val"].append(val_precision)
        classification_metrics_dict["Recall"]["val"].append(val_recall)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}  Val Accuracy: {val_accuracy:.4f}")
        print(f"Train F1 Score: {train_f1:.4f}  Val F1 Score: {val_f1:.4f}")
        print(f"Train Precision: {train_precision:.4f}  Val Precision: {val_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f}  Val Recall: {val_recall:.4f}")
        print("------------------------------------")

        # Check for early stopping
        if best_val_auc > classification_metrics_dict["Accuracy"]["val"][-1]:
            best_val_counter += 1
            if best_val_counter == patience:
                print(f'Early stopping. No improvement in validation loss for {patience} consecutive epochs.')
                break
        else:
            best_val_counter = 0

    return classification_metrics_dict


num_epochs = 40
classification_metric_names = ["Loss", "Accuracy", "F1 Score", "Precision", "Recall"]
for model_name, model in classification_models.items():
    print(f'Model: {model_name}')
    
    model, criterion, optimizer, scheduler = modify_model(model_name, model)
    # Train and validate the model
    classification_metrics_dict = train_and_validate_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler)

    # Plot metrics
    classification_plot_title = f"Training and Validation Metrics for {model_name}"
    plot_metrics(classification_metrics_dict, classification_metric_names, classification_plot_title)





