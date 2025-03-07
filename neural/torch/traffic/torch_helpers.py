import os

import cv2
from alive_progress import alive_bar
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf


class LabelDirImageDataset(Dataset):
    def __init__(self, data_dir, convert_input=None):
        print(f"Investigating {data_dir}...")
        self.img_labels = []
        labels = os.listdir(data_dir)
        for label in labels:
            dir = os.path.join(data_dir, label)
            for file_name in os.listdir(dir):
                self.img_labels.append((os.path.join(label, file_name), int(label)))
        print(f"Found {len(self.img_labels)} images in {len(labels)} label directories")
        self.img_dir = data_dir
        self.convert_input = convert_input
        
        self.tensor_cache = dict()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if idx in self.tensor_cache:
            return self.tensor_cache[idx]
        
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        label = self.img_labels[idx][1]
        image = cv2.imread(img_path)
        if self.convert_input:
            image = self.convert_input(image)

        tensor = tf.to_tensor(image)
        self.tensor_cache[idx] = (tensor, label)
        
        return (tensor, label)


class ModelRunner(ABC):
    def __init__(self, config):
        self.device = config["device"]
        self.model = config["model"]
        self.criterion = config["criterion"]
        self.optimizer = config["optimizer"]
        print(f"ModelRunner {self.__class__.__name__} initialized with device: {self.device}")
    
    def run_model(self, loader):
        self.setup(loader)
        with alive_bar(len(loader) * loader.batch_size) as bar:
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                loss = self.process(inputs, labels)
                bar.text(f"loss: {loss.item()}")
                bar(loader.batch_size)
                
        return self.get_results()
    
    @abstractmethod
    def setup(self, loader):
        pass
    
    @abstractmethod
    def process(self, inputs, labels):
        pass
    
    @abstractmethod
    def get_results(self):
        pass
    
    
class TrainingRunner(ModelRunner):
    def setup(self, loader):
        self.model.train()
        
    def process(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self._last_loss = loss
        return loss
    
    def get_results(self):
        return self._last_loss
    

class TestRunner(ModelRunner):
    def __init__(self, config):
        super().__init__(config)
        self.total_loss = 0
        self.correct = 0
        self.total = 0
        
    def setup(self, loader):
        self._num_batches = len(loader)
        self.model.eval()
        
    def process(self, inputs, labels):
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            self.total_loss += loss.item()
            self.total += labels.size(0)
            self.correct += (predicted == labels).sum().item()
        return loss
    
    def get_results(self):
        return self.total_loss / self._num_batches, self.correct / self.total
