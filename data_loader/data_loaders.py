import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
import pandas as pd
from PIL import Image
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import torch


class IsicDataSet(Dataset):
    def __init__(self, data_dir, split, training_split, validation_split, transform, is_transform):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.validation_split = validation_split
        self.training_split = training_split
        self.is_transform = is_transform
        self.train_image_path = os.path.join(self.data_dir, "ISIC-2019/ISIC_2019_Training_Input")
        self.test_image_path = os.path.join(self.data_dir, "ISIC-2019/ISIC_2019_Test_Input")
        self.train_gt_path = os.path.join(self.data_dir, "ISIC-2019/ISIC_2019_Training_GroundTruth.csv")
        self.train_metadata_gt_path = os.path.join(self.data_dir, "ISIC-2019/ISIC_2019_training_Metadata.csv")

        # Load the CSV file into a pandas DataFrame
        self.train_gt_pdf = pd.read_csv(self.train_gt_path)
        self.train_gt_pdf.drop(columns=['UNK'], inplace=True)
        self.train_image_name = self.train_gt_pdf["image"]
        self.train_label_list = self.train_gt_pdf.iloc[:, 1:].apply(
            lambda row: {col.lower(): row[col] for col in self.train_gt_pdf.columns[1:]}, axis=1).tolist()

        self.class_distribution = self.calculate_class_distribution()
        # Calculate the number of samples for each split
        self.total_samples = int(sum(self.class_distribution.values()))
        # Create indices array
        all_indices = np.arange(self.total_samples)

        # Get the sampled indices from the train_sampler

        self.train_size = int(self.training_split * self.total_samples)
        self.train_indices = np.array(all_indices[:self.train_size])
        self.valid_size = int(self.validation_split * (self.total_samples))
        self.test_size = self.total_samples - self.train_size - self.valid_size

        # Assign weights to each sample based on its class frequency
        class_frequencies = [self.class_distribution[class_name] for class_name in self.class_distribution]
        self.weights = [1.0 / class_freq for class_freq in class_frequencies]
        self.train_data, self.val_data, self.test_data = self.split_datasets()

    def __len__(self):
        return len(self.train_gt_pdf)

    def __getitem__(self, index):
        df = self.train_gt_pdf
        # Load image using self.df['image'][index], assuming 'image' is the column containing image paths
        image_path = os.path.join(self.train_image_path, df['image'][index] + ".jpg")
        # Replacing backslashes with forward slashes
        image_path = image_path.replace("\\", "/")
        image = Image.open(image_path).convert('RGB')  # Adjust as needed
        # Apply transformations if specified
        if self.is_transform:
            image = self.transform(image)

        # Extract class labels, assuming 'MEL', 'NV', etc., are columns in your CSV file
        labels = df.iloc[index, 1:].values.astype(float)
        labels = torch.tensor(labels)
        return image, labels

    def calculate_class_distribution(self):
        if self.split == "train":
            class_distribution = {
                label: 0 for item in self.train_label_list for label in item.keys()}

        for item in self.train_label_list:
            for label, count in item.items():
                class_distribution[label] += count

        return class_distribution

    def split_datasets(self):
        # List to store the training DataFrames
        train_dfs = []
        val_dfs = []
        # Function to return a random sample of 70% data from each class

        def sample_data(group, ratio):
            return group.sample(frac=ratio)

        # Loop through each class column
        for class_column in self.train_gt_pdf.columns[1:]:
            # Extract the current class label
            current_class = class_column

            # Filter rows for the current class
            class_df = self.train_gt_pdf[self.train_gt_pdf[current_class] == 1.0]

            # Sample 70% of the data for the current class
            sampled_train_df = class_df.groupby(current_class).apply(sample_data, ratio=self.training_split)

            train_dfs.append(sampled_train_df)

        train_data = pd.concat(train_dfs)
        remain_df = self.train_gt_pdf[~self.train_gt_pdf['image'].isin(train_data['image'])]
        # Add the following assertion
        assert len(remain_df) + len(train_data) == len(
            self.train_gt_pdf), "Mismatch in lengths of remain_df and train_data with original ground truth!"
        # Loop through each class column
        for class_column in remain_df.columns[1:]:
            # Extract the current class label
            current_class = class_column

            # Filter rows for the current class
            class_df = remain_df[remain_df[current_class] == 1.0]

            # Sample 70% of the data for the current class
            sampled_valid_df = class_df.groupby(current_class).apply(sample_data, ratio=self.training_split)

            val_dfs.append(sampled_valid_df)

        val_data = pd.concat(val_dfs)

        test_data = self.train_gt_pdf[~self.train_gt_pdf['image'].isin(
            train_data['image']) & ~self.train_gt_pdf['image'].isin(val_data['image'])]

        assert len(train_data) + len(val_data) + len(test_data) == len(
            self.train_gt_pdf), "Mismatch in lengths of remain_df and train_data and val_data with original ground truth!"
        assert len(self.train_gt_pdf['image'].unique()) == len(self.train_gt_pdf), "Duplicate instances in train_data!"
        assert len(train_data['image'].unique()) == len(train_data), "Duplicate instances in train_data!"

        # Check if values in the "image" column of val_data are unique
        assert len(val_data['image'].unique()) == len(val_data), "Duplicate instances in val_data!"

        # Check if values in the "image" column of test_data are unique
        assert len(test_data['image'].unique()) == len(test_data), "Duplicate instances in test_data!"

        # Convert the "image" columns to sets
        train_images_set = set(train_data['image'])
        val_images_set = set(val_data['image'])
        test_images_set = set(test_data['image'])

        # Check for overlap using set operations
        assert not train_images_set.intersection(val_images_set), "Overlap between train_data and val_data!"
        assert not train_images_set.intersection(test_images_set), "Overlap between train_data and test_data!"
        assert not val_images_set.intersection(test_images_set), "Overlap between val_data and test_data!"
        # # Save DataFrames to CSV files
        # train_data.to_csv('train_data.csv', index=False)
        # val_data.to_csv('val_data.csv', index=False)
        # test_data.to_csv('test_data.csv', index=False)
        # # Save DataFrames to pickle files
        # train_data.to_pickle('train_data.pkl')
        # val_data.to_pickle('val_data.pkl')
        # test_data.to_pickle('test_data.pkl')
        # Reset the index
        # train_data.reset_index(drop=True, inplace=True)
        # val_data.reset_index(drop=True, inplace=True)
        # test_data.reset_index(drop=True, inplace=True)

        # Check if values in the "image" column of train_data are unique

        return train_data, val_data, test_data


class IsicDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, training_split=0.7, validation_split=0.15, num_workers=1, split="train", is_transform=True):
        trsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.dataset = IsicDataSet(data_dir=data_dir, split=split, training_split=training_split,
                                   validation_split=validation_split, transform=trsfm, is_transform=is_transform)
        # self.sampler, self.valid_sampler, self.test_sampler = self.get_valid_test_sampler()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        weights = self.dataset.weights

        # Create WeightedRandomSampler for training
        train_sampler = WeightedRandomSampler(weights, num_samples=self.batch_size, replacement=True)

        # Get the sampled indices from the train_sampler
        train_indices = list(train_sampler)

        # Use the remaining samples for validation and test
        remaining_indices = list(range(self.dataset.total_samples))
        remaining_indices = [index for index in remaining_indices if index not in train_indices]

        # Split the remaining indices into validation and test sets
        valid_indices, test_indices = train_test_split(
            remaining_indices, train_size=self.dataset.valid_size, shuffle=False)

        # Create SubsetRandomSampler for validation and SequentialSampler for test
        valid_sampler = SubsetRandomSampler(valid_indices)
        self.test_sampler = SequentialSampler(test_indices)

        return train_sampler, valid_sampler,

    def split_validation(self):
        valid_data_loader = DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        test_data_loader = DataLoader(sampler=self.test_sampler, **self.init_kwargs)
        return valid_data_loader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, ):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True,
                                      transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
