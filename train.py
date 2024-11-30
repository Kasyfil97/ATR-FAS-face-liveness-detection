import tqdm
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from dataclasses import dataclass

from model.model import ATRFAS

#########===================================== set config ===============================#########
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)

@dataclass
class TRAINCONFIG:
    num_epochs: int = 10
    num_fold: int = 5
    model_save_path: str = 'model.pth'
    eval_per_epoch: int = 1
    gradient_accumulation: int = 1
    lambda_param: tuple = (0.5, 0.3, 0.2)


#########===================================== dataset ===============================#########

class DepthNormalize:
    def __call__(self, tensor: torch.Tensor):
        tensor = tensor/255.
        return tensor

class ATRFASDataset(Dataset):
    def __init__(self, data: pd.DataFrame, number_attack: int=3):
        self.data = data
        self.M = number_attack
        self.input_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # DepthNormalize()
        ])

    def __len__(self):
        return len(self.data)
    
    def open_images(self, folder, is_input: bool=True)->torch.Tensor:
        images = []
        filenames = os.listdir(folder)
        required_input = 6 if is_input else 5
        if len(filenames) != required_input:
            raise ValueError(f"Expected {required_input} images, but got {len(filenames)} images")
        
        for filename in os.listdir(folder):
            try:
                image = Image.open(os.path.join(folder, filename))
                if is_input:
                    image = self.input_transform(image)
                else:
                    image = self.depth_transform(image)
                images.append(image)
            except Exception as e:
                raise ValueError(f"Error loading image on {filename}: {e}")
        images = torch.stack(images).float()
        return images
    
    def  __getitem__(self, index) -> tuple:
        input_path = self.data.iloc[index]['input_path']
        depth_path = self.data.iloc[index]['depth_path']
        class_label = self.data.iloc[index]['label']

        inputs = self.open_images(input_path) # [6, 3, 256, 256]
        depths = self.open_images(depth_path, is_input=False) # [6, 1, 64, 64]
        if class_label==1:
            gate_label = torch.ones(3)
        else:
            gate_label = torch.zeros(3)
        return inputs, depths, gate_label, class_label

#########===================================== trainer ===============================#########

class Trainer:
    def __init__(self, model, criterion, optimizer, lambda_param, scheduler=None, device='cuda'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history={'train_loss': [], 'train_far': [], 'train_frr': [], 'val_loss': [], 'val_far': [], 'val_frr': []}
        self.lambda_param = lambda_param

    @staticmethod
    def accuracy(y_true, y_pred, threshold: float=0.5):
        y_pred_binary = (y_pred >= threshold).astype(int)
        tn, fp, fn, tp=confusion_matrix(y_true, y_pred_binary)
        far = fp(fp+tn)
        frr = fn/(fn+tp)
        return far, frr
    
    @staticmethod
    def loss_depth(ground_truth: torch.Tensor, pred: torch.Tensor):
        mean_gt = ground_truth.mean(dim=1)     # (batch_size, 64, 64)
        mean_pred = pred.mean(dim=1)           # (batch_size, 64, 64)
        
        # Flatten spatial dimensions
        mean_gt = mean_gt.view(mean_gt.shape[0], -1)      # (batch_size, 64*64)
        mean_pred = mean_pred.view(mean_pred.shape[0], -1) # (batch_size, 64*64)
        
        # Normalize ground truth to sum to 1 for each sample
        mean_gt = mean_gt / (mean_gt.sum(dim=1, keepdim=True) + 1e-5)
        
        F.cross_entropy(mean_pred, mean_gt)
        
        # Average over batch
        loss = loss.mean()
        
        return loss

    @staticmethod
    def loss_gate(ground_truth: torch.Tensor, pred: torch.Tensor):
        return F.cross_entropy(pred, ground_truth).mean()

    @staticmethod
    def loss_classification(ground_truth: torch.Tensor, pred: torch.Tensor):
        return F.binary_cross_entropy(pred, ground_truth).mean()
    
    def train_step(self, train_loader: DataLoader, gradient_accumulation: int)->tuple:
        epoch_loss = 0.0
        epoch_far = 0.0
        epoch_frr = 0.0
        
        self.model.train()
        for i, (x, depths,gate_label, y) in enumerate(tqdm(train_loader, total=len(train_loader))):
            if self.device=='cuda':
                x, depths,gate_label, y=x.to(self.device), depths.to(self.device), gate_label.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            pred, type_gating, frame_depth_map, frame_attention_map, depth_map = self.model(x, infer_type='training')

            # calculate loss
            loss_gate = Trainer.loss_gate(gate_label, type_gating)
            loss_depth = Trainer.loss_depth(depths, frame_depth_map)
            loss_classification = Trainer.loss_classification(y, pred)
            loss = self.lambda_param[0] * loss_gate + self.lambda_param[1] * loss_depth + self.lambda_param[2] * loss_classification

            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation #loss should devide to number of gradient of accumulation
            loss.backward()
            if (i+1)%gradient_accumulation==0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                

            y_pred = pred.argmax(dim=1)
            far, frr = Trainer.accuracy(y, y_pred.cpu().numpy())

            epoch_loss+=loss.item()*gradient_accumulation
            epoch_far+=far
            epoch_frr+=frr
        epoch_loss/=len(train_loader)
        epoch_far/=len(train_loader)
        epoch_frr/=len(train_loader)
        
        return epoch_loss, epoch_far, epoch_frr

    def eval_step(self, val_loader: Dataset)->tuple:
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        self.model.eval()
        for x, depths,gate_label, y in tqdm(val_loader, total=len(val_loader)):
            if self.device=='cuda':
                x, depths,gate_label, y=x.to(self.device), depths.to(self.device), gate_label.to(self.device), y.to(self.device)

            with torch.no_grad():
                pred, type_gating, frame_depth_map, frame_attention_map, depth_map = self.model(x, infer_type='training')
                # calculate loss
                loss_gate = Trainer.loss_gate(gate_label, type_gating)
                loss_depth = Trainer.loss_depth(depths, frame_depth_map)
                loss_classification = Trainer.loss_classification(y, pred)
                loss = self.lambda_param[0] * loss_gate + self.lambda_param[1] * loss_depth + self.lambda_param[2] * loss_classification

            y_pred = pred.argmax(dim=1)
            far, frr = Trainer.accuracy(y, y_pred.cpu().numpy())
            epoch_loss+=loss.item()
            epoch_far+=far
            epoch_frr+=frr
        epoch_loss/=len(val_loader)
        epoch_far/=len(val_loader)
        epoch_frr/=len(val_loader)
        
        return epoch_loss, epoch_far, epoch_frr

    def fit(self, 
            dataset: pd.DataFrame, 
            num_epochs: int,
            num_fold: int,
            lambda_param: dict,
            model_save_path: str,
            eval_per_epoch: int=1, 
            gradient_accumulation: int=1):
        skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset['target'])):
            print(f"================Fold {fold + 1}/{num_fold}===============")

            # split train and val based on fold
            train = dataset.iloc[train_idx].reset_index(drop=True)
            val = dataset.iloc[train_idx].reset_index(drop=True)

            # create dataloader
            trainDataset = ATRFASDataset(train)
            valDataset = ATRFASDataset(val)

            trainLoader = DataLoader(
                trainDataset,
                batch_size=32,
                shuffle=True,
                drop_last=True,
                num_workers=2
            )
            valLoader = DataLoader(
                valDataset,
                batch_size=32,
                drop_last=False,
                num_workers=2
            )

            
            for epoch in range(num_epochs):
                loss, far, frr = self.train_step(trainLoader, 
                                            gradient_accumulation=gradient_accumulation, 
                                            lambda_param=lambda_param)
                self.history['train_loss'].append(loss)
                self.history['train_far'].append(far.cpu().item())
                self.history['train_frr'].append(frr.cpu().item())
    
                print(f'EPOCH: {epoch+1}/{num_epochs}: train loss: {loss:.4f}, train FAR: {far:.4f}, train FRR: {frr:.4f}')
                if epoch%eval_per_epoch==0 and eval_per_epoch<=num_epochs:
                    loss, far, frr = self.eval_step(valLoader)
    
                    self.history['val_loss'].append(loss.cpu().item())
                    self.history['val_far'].append(far.cpu().item())
                    self.history['val_frr'].append(frr.cpu().item())
                    print(f'Validation Loss: {loss:.4f}, Validation FAR: {far:.4f}, Validation FRR: {frr:.4f}')
    
                if self.scheduler:
                    self.scheduler.step()


        self.model.save(model_save_path)

#########===================================== main ===============================#########
if __name__ == "__main__":
    seed_everything(42)
    train_config = TRAINCONFIG()

    # load dataset
    dataset = pd.read_csv('dataset.csv')

    model = ATRFAS(num_batches=32, num_frames=6, return_proba=True, infer_type='inference')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    trainer = Trainer(model, criterion, optimizer, train_config.lambda_param, scheduler)
    trainer.fit(dataset, train_config.num_epochs, train_config.num_fold, train_config.lambda_param, train_config.model_save_path, train_config.eval_per_epoch, train_config.gradient_accumulation)
    print("Training finished")
    print("Saving model...")
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved")