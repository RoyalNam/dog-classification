import torchvision
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.datasets import ImageFolder
from cnnClassifier.entity.config_entity import DataLoaderConfig
from cnnClassifier.utils.common import save_model


class Dataloader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config

    def create_dataloader(self):
        weights = torchvision.models.get_weight(f"{self.config.weights}.DEFAULT")
        transforms = weights.transforms()
        train_data = ImageFolder(
            root=str(self.config.train_dir),
            transform=transforms
        )
        test_data = ImageFolder(
            root=str(self.config.test_dir),
            transform=transforms
        )

        classes_name = train_data.classes

        train_dataloader = TorchDataLoader(
            dataset=train_data,
            batch_size=self.config.params_batch_size,
            shuffle=True
        )
        test_dataloader = TorchDataLoader(
            dataset=test_data,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )
        # save transforms image
        save_model(transforms, self.config.image_transforms_path)
        self.save_classes(classes_name)
        return train_dataloader, test_dataloader, classes_name

    @staticmethod
    def save_classes(classes_name):
        with open('classes.txt', 'w') as file:
            for breed in classes_name:
                file.write(f"{breed}\n")
                