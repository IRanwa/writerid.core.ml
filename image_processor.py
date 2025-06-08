from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.grayscale_mean = [0.5]
        self.grayscale_std = [0.5]

    def get_train_transform(self) -> transforms.Compose:
        transform_list = [
            transforms.RandomResizedCrop(self.image_size),
            transforms.ToTensor(),
        ]
        transform_list.append(transforms.Normalize(mean=self.grayscale_mean, std=self.grayscale_std))
        return transforms.Compose(transform_list)

    def get_test_transform(self) -> transforms.Compose:
        transform_list = [
            transforms.Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)]),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ]
        transform_list.append(transforms.Normalize(mean=self.grayscale_mean, std=self.grayscale_std))
        return transforms.Compose(transform_list)