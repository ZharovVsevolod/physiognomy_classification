import os
import pickle
from torchvision import transforms
from physiognomy_classification.data import ChalearnDataset, format_label_list

def get_data():
    # Set data related paths
    train_data_dir = "dataset/train/faces"
    valid_data_dir = "dataset/val/faces"
    test_data_dir = "dataset/test/faces"
    train_labels_pickle = "dataset/train/annotation_training.pkl"
    valid_labels_pickle = "dataset/val/annotation_validation.pkl"
    test_labels_pickle = "dataset/test/annotation_test.pkl"
    
    # Load label pickle files
    train_annotations = pickle.load(open(train_labels_pickle, "rb" ), encoding="latin1" )
    val_annotations = pickle.load(open(valid_labels_pickle, "rb" ), encoding="latin1")
    test_annotations = pickle.load(open(test_labels_pickle, "rb" ), encoding="latin1")
    
    # Normalise data and define data augmentations
    train_transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    test_transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    train_annotations = format_label_list(train_annotations)
    val_annotations = format_label_list(val_annotations)
    test_annotations = format_label_list(test_annotations)
    trainset = ChalearnDataset(train_annotations, train_data_dir, transform=train_transform)
    valset = ChalearnDataset(val_annotations, valid_data_dir, transform=test_transform)
    testset = ChalearnDataset(test_annotations, test_data_dir, transform=test_transform)

    print("Dataset is ready")
    return trainset, valset, testset

if __name__ == "__main__":
    trainset, valset, testset = get_data()
    
    print(len(trainset), "samples in training dataset.")
    print(len(valset), "samples in validation dataset.")
    print(len(testset), "samples in test dataset.")

    print("-----")

    gt = trainset[1702]

    print(gt[0].shape)
    print(gt)