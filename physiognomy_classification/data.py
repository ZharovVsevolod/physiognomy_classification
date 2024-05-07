import os
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
import lightning as L
from torchvision import transforms
import pickle

# Helper function to format the label files such that filenames are primary key values of a dictionary
# that contains the corresponding big five scores for each file
def format_label_list(label_dict):
    # IMPORTANT: LABEL ORDER IS 'O-C-E-A-N'
    label_keys_order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    new_label_dict = {}
    for key in label_keys_order:
        # Remove unused 'interview' label from the .pkl files given by Chalearn
        if key == 'interview':
            continue;
        for sample in label_dict[key]:
            if sample not in new_label_dict:
                new_label_dict[sample] = [label_dict[key][sample]]
            else:
                new_label_dict[sample].append(label_dict[key][sample])
    return new_label_dict

class ChalearnDataset(Dataset):
    """ChaLearn First Impression Dataset"""
    def __init__(self, label_dict, root_dir, transform=None):
        """
        Args:
            label_dict (dictionary): Dictionary with (original) filenames as keys and big 5
                label dictionaries {extraversion: 0.3842833, neuroticism: 0.742332} as values.
            root_dir (string): Directory with all the images.
                Original filename can be retrieved as
                label_key = img_name[:15] + '.mp4'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = label_dict
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = os.listdir(root_dir)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.image_names[idx]
        
        # Load image
        img_path = self.root_dir + "/" + img_name
        image = io.imread(img_path) / 255
        image = torch.Tensor(image).float()
        image = image.reshape(3, 224, 224)
        
        # Get the labels
        label_key = img_name[:15] + '.mp4'
        big_five_scores = self.labels[label_key]
        big_five_scores = torch.FloatTensor(big_five_scores)
        
        if self.transform:
            image = self.transform(image)
        
        return image, big_five_scores


class ChalearnDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_data_dir:str,
            valid_data_dir:str,
            test_data_dir:str,
            train_labels_pickle:str,
            valid_labels_pickle:str,
            test_labels_pickle:str,
            batch_size:int
        ) -> None:
        super().__init__()
        self.train_data_dir = train_data_dir
        self.valid_data_dir = valid_data_dir
        self.test_data_dir = test_data_dir
        self.train_labels_pickle = train_labels_pickle
        self.valid_labels_pickle = valid_labels_pickle
        self.test_labels_pickle = test_labels_pickle

        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.test_transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            train_annotations = pickle.load(open(self.train_labels_pickle, "rb" ), encoding="latin1" )
            val_annotations = pickle.load(open(self.valid_labels_pickle, "rb" ), encoding="latin1")
            train_annotations = format_label_list(train_annotations)
            val_annotations = format_label_list(val_annotations)

            self.train_dataset = ChalearnDataset(
                label_dict = train_annotations, 
                root_dir = self.train_data_dir, 
                transform = self.train_transform
            )
            self.val_dataset = ChalearnDataset(
                label_dict = val_annotations, 
                root_dir = self.valid_data_dir, 
                transform = self.test_transform
            )
            print("Stage `fit` is set")
            print(f"Train len is {len(self.train_dataset)}")
            print(f"Val len is {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            test_annotations = pickle.load(open(self.test_labels_pickle, "rb" ), encoding="latin1")
            test_annotations = format_label_list(test_annotations)
            self.test_dataset = ChalearnDataset(
                label_dict = test_annotations, 
                root_dir = self.test_data_dir, 
                transform = self.test_transform
            )
            print("Stage `test` is set")
            print(f"Test len is {len(self.test_dataset)}")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )