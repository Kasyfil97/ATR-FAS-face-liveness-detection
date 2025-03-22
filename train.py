# v3.0.0
import os
import glob
import cv2
import wandb
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dataclasses import dataclass
from google.cloud import storage

from model.model import ATRFAS

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

#########===================================== set seed ===============================#########
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)


# ########===================================== dataset ===============================#########

class DepthNormalize:
    def __call__(self, tensor: torch.Tensor):
        tensor = tensor/255.
        return tensor

class ATRFASDataset(Dataset):
    def __init__(self, data: pd.DataFrame, is_train: bool, number_attack: int=3):
        self.data = data
        self.M = number_attack
        self.is_train = is_train
        
        self.basic_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # DepthNormalize()
        ])
        # Augmentation transformations with low intensity
        self.augmentations = transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3))  # Light blur
            ], p=0.3),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=(0.8, 1.3))  # Subtle brightness adjustment
            ], p=0.4),
            transforms.RandomApply([
                transforms.ColorJitter(hue=(-0.05,0))  # Subtle hue adjustment
            ], p=0.4),
            transforms.RandomApply([
                transforms.ColorJitter(saturation=(0.8,1.3))  # Subtle saturation adjustment
            ], p=0.4),
            transforms.RandomApply([
                transforms.RandomPosterize(bits=7, p=0.3)  # Light posterization
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(2, p=1)  # Light posterization
            ], p=0.3),
        ])
        
        self.differential_matrix= torch.tensor(
        [
            [1, -1, 0, 0, 0],
            [1, 0, -1, 0, 0],
            [1, 0, 0, -1, 0],
            [0, 1, 0, 0, -1],
            [0, 0, 1, 0, -1],
            [0, 0, 0, 1, -1]
            ], dtype=torch.float32
            )
        
    def train_transform(self, image):
        # Apply augmentations
        image = self.augmentations(image)
        image = self.basic_transform(image)
        
        return image

    def __len__(self):
        return len(self.data)
    
    def open_images(self, folder,  is_input=True)->torch.Tensor:
        images = []
        filenames = sorted([image for image in os.listdir(folder) if image.endswith(('.jpg','.png'))])
        required_input = 5
        if len(filenames) != required_input:
            raise ValueError(f"Expected {required_input} images, but got {len(filenames)} images")
        
        for filename in sorted(filenames):
            try:
                if is_input:
                    image = cv2.imread(os.path.join(folder, filename))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    image = Image.fromarray(image)
                    if self.is_train:
                        image = self.train_transform(image)
                    else:
                        image = self.basic_transform(image)
                else:
                    image = Image.open(os.path.join(folder, filename))
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
        type_image = self.data.iloc[index]['type']
        label = torch.nn.functional.one_hot(torch.tensor(class_label), num_classes=2).float()

        images = self.open_images(input_path) # [5, 3, 256, 256]
        input_diff = torch.tensordot(self.differential_matrix, images, dims=1).float() #[6, 3, 256,256]
        
        depth = self.open_images(depth_path, is_input=False) # [6, 1, 64, 64]
        type_to_gate_label = {
            'real': torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32),
            'paper': torch.tensor([1, 0, 0], dtype=torch.float32),
            'screen': torch.tensor([0, 1, 0], dtype=torch.float32),
            'mask': torch.tensor([0, 0, 1], dtype=torch.float32)
        }

        if type_image not in type_to_gate_label:
            raise ValueError(f"Unknown type image {type_image}")

        gate_label = type_to_gate_label[type_image]
        return input_diff, depth, gate_label, label

#########================================== upload model =============================#########
class UploadGCS:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def upload(self, source_file_name: str, destination_blob_name: str):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

# ########===================================== trainer ===============================#########

class Trainer:
    def __init__(self, 
                 model,
                 optimizer,
                 lambda_param, 
                 model_ver='v1',
                 scheduler=None,
                 checkpoint_dir=None,
                 save_total_limit=1,
                 bucket_name = None, # if you want to save model to GCS
                 device='cuda'):
        
        self.model = model.to(device)
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else 'checkpoint'
        self.optimizer = optimizer
        self.lambda_param = lambda_param
        self.model_ver = model_ver
        self.scheduler = scheduler
        self.save_total_limit = save_total_limit
        self.device = device
        self.history={'train_loss': {'total':[], 'gate':[],'classification':[],'depth':[]}, 
                      'train_acc': {'far':[],'frr':[],'precision':[],'recall':[], 'accuracy':[]}, 
                      'val_loss': {'total':[], 'gate':[],'classification':[],'depth':[]}, 
                      'val_acc': {'far':[],'frr':[],'precision':[],'recall':[], 'accuracy':[]}}
        self.upload_gcs = UploadGCS(bucket_name) if bucket_name else None 
        
        self.best_val_acc = float('inf')
        

    @staticmethod
    def accuracy(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        tn, fp, fn, tp=confusion_matrix(y_true, y_pred).ravel()
        far = 0 if (fp+tn)==0 else fp/(fp+tn)
        frr = 0 if (fn+tp)==0 else fn/(fn+tp)
        return far, frr, accuracy, precision, recall

    
    @staticmethod
    def loss_depth(ground_truth: torch.Tensor, pred: torch.Tensor):
        # groundtruth shape (batch size, 5, 1, 64, 64)
        # predict shape (batch size, 6, 1, 64, 64)
        mean_gt = ground_truth.mean(dim=1)     # (batch_size, 64, 64)
        mean_pred = pred.mean(dim=1)           # (batch_size, 64, 64)
        
        # Flatten spatial dimensions
        mean_gt = mean_gt.view(mean_gt.shape[0], -1)      # (batch_size, 64*64)
        mean_pred = mean_pred.view(mean_pred.shape[0], -1) # (batch_size, 64*64)
        
        epsilon = 1e-7
        mean_pred = torch.clamp(mean_pred, min=epsilon, max=1.0)
        loss = -torch.sum(mean_gt * torch.log(mean_pred),dim=1)/(mean_gt.shape[1])
        
        return loss.mean()

    @staticmethod
    def loss_gate(ground_truth: torch.Tensor, pred: torch.Tensor):
        """ calculate the gate loss"""
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, ground_truth)
        return loss
    # @staticmethod
    

    @staticmethod
    def loss_classification(ground_truth: torch.Tensor, pred: torch.Tensor):
        """ calculate the classification loss"""
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, ground_truth)
        return loss

    def train_step(self, train_loader: DataLoader, gradient_accumulation: int, epoch: int)->tuple:
        epoch_loss = {'total': 0.0, 'gate': 0.0, 'depth': 0.0, 'classification': 0.0}
        total_grad_norm = 0.0
        
        y_true = []
        y_pred = []
        self.model.train()
        for i, (x, depths,gate_label, y) in enumerate(tqdm(train_loader, total=len(train_loader), desc=f'Training {epoch+1}')):
            if self.device=='cuda':
                x, depths,gate_label, y=x.to(self.device), depths.to(self.device), gate_label.to(self.device), y.to(self.device)
                
                
            pred, type_gating, frame_depth_map, frame_attention_map, depth_map = self.model(x)

            # calculate loss
            loss_gate = Trainer.loss_gate(gate_label, type_gating)
            loss_depth = Trainer.loss_depth(depths, frame_depth_map)
            loss_classification = Trainer.loss_classification(y, pred)
            loss = self.lambda_param[0] * loss_gate + self.lambda_param[1] * loss_depth + self.lambda_param[2] * loss_classification
            
            loss /= gradient_accumulation
            loss.backward()
            
            if (i + 1) % gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                total_grad_norm += grad_norm.item()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Log the gradient norm
                wandb.log({
                    "Train_Loss/Grad_Norm": grad_norm.item(),
                })
            
            epoch_loss['total']+=loss.item()*gradient_accumulation
            epoch_loss['gate']+=loss_gate.item()
            epoch_loss['depth']+=loss_depth.item()
            epoch_loss['classification']+=loss_classification.item()

            y_pred.append(pred.argmax(dim=1).detach())
            y_true.append(y.argmax(dim=1).detach())
            
        if len(train_loader) % gradient_accumulation != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        far, frr, accuracy, precision, recall = Trainer.accuracy(y_true, y_pred)

        epoch_acc = {
            'far': far,
            'frr': frr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        epoch_loss={k:v/len(train_loader) for k,v in epoch_loss.items()}
        
        return epoch_loss, epoch_acc

    def eval_step(self, val_loader: DataLoader)->tuple:
        epoch_loss = {'total': 0.0, 'gate': 0.0, 'depth': 0.0, 'classification': 0.0}

        y_true = []
        y_pred = []

        self.model.eval()
        for x, depths,gate_label, y in tqdm(val_loader, total=len(val_loader), desc='Validation'):
            if self.device=='cuda':
                x, depths,gate_label, y=x.to(self.device), depths.to(self.device), gate_label.to(self.device), y.to(self.device)

            with torch.no_grad():
                pred, type_gating, frame_depth_map, frame_attention_map, depth_map = self.model(x)
                # calculate loss
                loss_gate = Trainer.loss_gate(gate_label, type_gating)
                loss_depth = Trainer.loss_depth(depths, frame_depth_map)
                loss_classification = Trainer.loss_classification(y.float(), pred)
                loss = self.lambda_param[0] * loss_gate + self.lambda_param[1] * loss_depth + self.lambda_param[2] * loss_classification

            epoch_loss['total']+=loss.item()
            epoch_loss['gate']+=loss_gate.item()
            epoch_loss['depth']+=loss_depth.item()
            epoch_loss['classification']+=loss_classification.item()

            y_pred.append(pred.argmax(dim=1))
            y_true.append(y.argmax(dim=1))
        
        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        
        far, frr, accuracy, precision, recall = Trainer.accuracy(y_true, y_pred)

        epoch_acc = {
            'far': far,
            'frr': frr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        epoch_loss={k:v/len(val_loader) for k,v in epoch_loss.items()}
        return epoch_loss, epoch_acc
    
    def fit(self, 
            train_loader,
            val_loader,
            start_from_last_ckpt = False, # True if you want start from the last checkpoint model, else False
            num_epochs = 10,
            eval_per_epoch =1, 
            gradient_accumulation =1):
        
        if start_from_last_ckpt:
            
            assert os.path.exists(self.checkpoint_dir), "The previous checkpoints does't exist, if you want start from scratch, please set start_from_last_ckpt False"
            
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "*.pth"))
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            start_epoch = self.load_checkpoint(latest_checkpoint)
        else:
            start_epoch = 0
            
        
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = self.train_step(train_loader,
                                                    gradient_accumulation=gradient_accumulation,
                                                    epoch= epoch)
            
            # self.update_history('train', train_loss, train_acc)

            print(f'EPOCH: {epoch+1}/{num_epochs}')
            print(f"Train Loss - Total: {train_loss['total']:.4f}, Gate: {train_loss['gate']:.4f}, Depth: {train_loss['depth']:.4f}, Classification: {train_loss['classification']:.4f}")
            print(f"Train Acc - FAR: {train_acc['far']:.4f}, FRR: {train_acc['frr']:.4f}, Accuracy: {train_acc['accuracy']:.4f}, Precision: {train_acc['precision']:.4f}, Recall: {train_acc['recall']:.4f}")

            wandb.log({
                    "Learning_Rate": self.optimizer.param_groups[0]['lr'],
                    "Train_Loss/Total": train_loss['total'],
                    "Train_Loss/Gate": train_loss['gate'],
                    "Train_Loss/Depth": train_loss['depth'],
                    "Train_Loss/Classification": train_loss['classification'],
                    "Train_Acc/FAR": train_acc['far'],
                    "Train_Acc/FRR": train_acc['frr'],
                    "Train_Acc/Accuracy": train_acc['accuracy'],
                    "Train_Acc/Precision": train_acc['precision'],
                    "Train_Acc/Recall": train_acc['recall'],
            })

            if epoch%eval_per_epoch==0 and eval_per_epoch<=num_epochs:
                val_loss, val_acc = self.eval_step(val_loader)

                # self.update_history('val', val_loss, val_acc)

                print(f'Validation Loss - Total: {val_loss["total"]:.4f}, Gate: {val_loss["gate"]:.4f}, Depth: {val_loss["depth"]:.4f}, Classification: {val_loss["classification"]:.4f}')
                print(f'Validation Acc - FAR: {val_acc["far"]:.4f}, FRR: {val_acc["frr"]:.4f}, Accuracy: {val_acc["accuracy"]:.4f}, Precision: {val_acc["precision"]:.4f}, Recall: {val_acc["recall"]:.4f}')

                HTER= (val_acc['far'] + val_acc['frr']) / 2
                
                wandb.log({
                    "Val_Loss/Total": val_loss['total'],
                    "Val_Loss/Gate": val_loss['gate'],
                    "Val_Loss/Depth": val_loss['depth'],
                    "Val_Loss/Classification": val_loss['classification'],
                    "Val_Acc/FAR": val_acc['far'],
                    "Val_Acc/FRR": val_acc['frr'],
                    "Val_Acc/Precision": val_acc['precision'],
                    "Val_Acc/Recall": val_acc['recall'],
                    "Val_Acc/Accuracy": val_acc['accuracy'],
                    "Val_Acc/HTER": HTER
                })

            if self.scheduler:
                self.scheduler.step()

            self.save_checkpoint(HTER, epoch)

    def update_history(self, phase, loss, acc):
        """Update training/validation history"""
        for key, value in loss.items():
            self.history[f'{phase}_loss'][key].append(value)
        for key, value in acc.items():
            self.history[f'{phase}_acc'][key].append(value)

    def load_checkpoint(self, checkpoint_path):
        """Load the latest checkpoint if it exists and return the starting epoch."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.scheduler:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['hter']
        start_epoch = checkpoint['epoch'] + 1
        print('lr', self.optimizer.param_groups[0]['lr'])
        return start_epoch  # Return the starting epoch


    def save_checkpoint(self, hter, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if hter < self.best_val_acc:
            self.best_val_acc = hter
            model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': epoch,
                'hter':self.best_val_acc
            }, model_path)
            print(f'best model at epoch {epoch} with HTER {hter}')
        else:
            model_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict':self.scheduler.state_dict() if self.scheduler else None,
                'epoch': epoch,
                'hter':self.best_val_acc
            }, model_path)
        if self.upload_gcs:
            self.upload_gcs.upload(model_path, f'model/{self.model_ver}/{model_path.split("/")[-1]}')
        ckpt_dir = [os.path.join(self.checkpoint_dir,model_file) for model_file in os.listdir(self.checkpoint_dir) if model_file.startswith('model_epoch_')]
        if len(ckpt_dir) > self.save_total_limit:
            oldest_file = min(ckpt_dir, key=os.path.getctime)
            os.remove(oldest_file)

#########===================================== config ===============================#########
@dataclass
class DATASETCONFIG:
    batch_size: int = 4
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False

@dataclass
class TRAINCONFIG:
    checkpoint = 'your-dir-ckpt-name'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs: int = 1000
    gamma = 0.95
    learning_rate: float = 1e-5
    eval_per_epoch: int = 1
    gradient_accumulation: int = 4
    weight_decay = 1e-5
    lambda_param: tuple = (1, 1, 1)  # (gate, depth, classification)
    save_total_limit: int = 5 # number of saving checkpoint in local
    start_from_last_ckpt: bool= True
    bucket_name: str = 'your-bucket-name'
    model_ver = 'model version'
    parallel_computing = True

# ########===================================== main ===============================#########

if __name__ == "__main__":
    train_config = TRAINCONFIG()
    seed_everything(42)
    print(train_config)
    dataset_config = DATASETCONFIG()

    wandb.login(key="your wandb key", force=True)
    wandb.init(project="the project name", 
               resume=True,
               config={'epochs':train_config.num_epochs},
               id='project id',
               notes="notes for project",)

    # load dataset
    train = pd.read_csv('train dataset.csv')
    val = pd.read_csv('val dataset.csv')
    
    train.reset_index(drop=True, inplace=True)
    
    print(f'train: {train.shape}, val: {val.shape}')
    
    # create dataloader
    train_dataset = ATRFASDataset(train, is_train=True)
    val_dataset = ATRFASDataset(val, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        drop_last=dataset_config.drop_last,
        num_workers=dataset_config.num_workers,
        shuffle=dataset_config.shuffle
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config.batch_size,
        drop_last=False,
        num_workers=dataset_config.num_workers,
            )
    
    # define model, optimizer, scheduler
    model = ATRFAS(num_frames=6, return_proba=False, infer_type='training')
    if train_config.parallel_computing:
        model = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config.gamma)
    # scheduler = None
    
    #set Trianer object and begin training
    trainer = Trainer(model=model,
                      optimizer=optimizer, 
                      lambda_param=train_config.lambda_param,
                      model_ver = train_config.model_ver,
                      bucket_name = train_config.bucket_name,
                      scheduler=scheduler, 
                      device=train_config.device, 
                      save_total_limit=train_config.save_total_limit,
                      checkpoint_dir=train_config.checkpoint
                      )
    
    trainer.fit(train_loader=train_loader,
                val_loader=val_loader,
                start_from_last_ckpt = train_config.start_from_last_ckpt,
                num_epochs=train_config.num_epochs,
                eval_per_epoch=train_config.eval_per_epoch, 
                gradient_accumulation=train_config.gradient_accumulation)
    print("Training finished")
    