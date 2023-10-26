import cv2
import torch
import albumentations

class TRSynthDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, char2int):
        """
        Initialize the dataset with image paths and labels.

        :param data_file: pandas dataframe containing the image path and labels
        :param char2int: dictionary mapping every character to a unique integer
        """
        self.image_ids = list(data_file["images"])
        self.labels = list(data_file["labels"])
        self.char2int = char2int
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.aug = albumentations.Compose(
            [albumentations.Normalize(mean, std,
                                      max_pixel_value=255.0,
                                      always_apply=True)]
        )
        # Find the maximum label length in the dataset
        self.max_len = data_file["labels"].apply(lambda x: len(x)).max()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image id
        img_id = self.image_ids[idx]
        # Load the image
        img = cv2.imread(img_id)
        # Normalize the image
        augmented = self.aug(image=img)
        img = augmented['image']
        # Bring channel first
        img = img.transpose(2, 0, 1)
        # Convert to torch tensor
        img = torch.from_numpy(img)

        # Get the labels
        target = self.labels[idx]
        # Convert characters to integers
        target = [self.char2int[i] for i in target]
        # Length of each target
        target_len = torch.LongTensor([len(target)])
        # Pad target with zeros to make sure every
        # target has an equal length
        target += [0] * (self.max_len - len(target))
        # Convert to torch tensor
        target = torch.LongTensor(target)

        return img, target, target_len
