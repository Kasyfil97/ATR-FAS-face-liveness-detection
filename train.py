import os
import wandb
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dataclasses import dataclass
from google.cloud import storage

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
    num_epochs: int = 2
    batch_size: int = 2
    num_fold: int = 2
    learning_rate: float = 1e-3
    eval_per_epoch: int = 1
    gradient_accumulation: int = 1
    lambda_param: tuple = (1, 1, 1)
    save_total_limit: int = 1


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
        type_image = self.data.iloc[index]['type']
        label = torch.nn.functional.one_hot(torch.tensor(class_label), num_classes=2)

        images = self.open_images(input_path) # [6, 3, 256, 256]
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
        return images, depth, gate_label, label

#########================================== upload model =============================#########
class UploadGCS:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def upload(self, source_file_name: str, destination_blob_name: str):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

#########===================================== trainer ===============================#########

class Trainer:
    def __init__(self, 
                 model,
                 optimizer, 
                 lambda_param, 
                 scheduler=None,
                 checkpoint_dir=None,
                 save_total_limit=1,
                 bucket_name = None, # if you want to save model to GCS
                 device='cuda'):
        
        self.model = model.to(device)
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else 'checkpoint'
        self.optimizer = optimizer
        self.lambda_param = lambda_param
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
        far = fp/(fp+tn)
        frr = fn/(fn+tp)
        return far, frr, accuracy, precision, recall

    
    @staticmethod
    def loss_depth(ground_truth: torch.Tensor, pred: torch.Tensor):
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
        return F.cross_entropy(pred, ground_truth)

    @staticmethod
    def loss_classification(ground_truth: torch.Tensor, pred: torch.Tensor):
        return F.binary_cross_entropy(pred, ground_truth)

    def train_step(self, train_loader: DataLoader, gradient_accumulation: int)->tuple:
        epoch_loss = {'total': 0.0, 'gate': 0.0, 'depth': 0.0, 'classification': 0.0}

        y_true = []
        y_pred = []
        self.model.train()
        for i, (x, depths,gate_label, y) in enumerate(tqdm(train_loader, total=len(train_loader), desc='Training')):
            if self.device=='cuda':
                x, depths,gate_label, y=x.to(self.device), depths.to(self.device), gate_label.to(self.device), y.to(self.device)

            if i % gradient_accumulation == 0:
                self.optimizer.zero_grad()
            pred, type_gating, frame_depth_map, frame_attention_map, depth_map = self.model(x)

            # calculate loss
            loss_gate = Trainer.loss_gate(gate_label, type_gating)
            loss_depth = Trainer.loss_depth(depths, frame_depth_map)
            loss_classification = Trainer.loss_classification(y.float(), pred)
            loss = self.lambda_param[0] * loss_gate + self.lambda_param[1] * loss_depth + self.lambda_param[2] * loss_classification
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation #loss should devide to number of gradient of accumulation
            loss.backward()
            if (i + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            epoch_loss['total']+=loss.item()*gradient_accumulation
            epoch_loss['gate']+=loss_gate.item()*gradient_accumulation
            epoch_loss['depth']+=loss_depth.item()*gradient_accumulation
            epoch_loss['classification']+=loss_classification.item()*gradient_accumulation

            y_pred.append(pred.argmax(dim=1).detach())
            y_true.append(y.argmax(dim=1).detach())
        
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
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

    def eval_step(self, val_loader: Dataset)->tuple:
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

            y_pred.append(pred.argmax(dim=1).detach())
            y_true.append(y.argmax(dim=1).detach())
        
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
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
            dataset,
            checkpoint_model = None, # if you want start from the last checkpoint model, input the path
            batch_size=1,
            num_epochs = 10,
            num_fold=5,
            num_workers=0,
            eval_per_epoch =1, 
            gradient_accumulation =1):
        
        start_epoch = 0 if not checkpoint_model else self.load_checkpoint(checkpoint_model)+1
        
        skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset['label'])):
            print(f"\n================Fold {fold + 1}/{num_fold}===============\n")

            # Reset model for new fold if not continuing from checkpoint
            if not checkpoint_model or fold > 0:
                self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.9)
                if self.scheduler:
                    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)

            # split train and val based on fold
            train = dataset.iloc[train_idx].reset_index(drop=True)
            val = dataset.iloc[val_idx].reset_index(drop=True)

            # create dataloader
            train_dataset = ATRFASDataset(train)
            val_dataset = ATRFASDataset(val)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                drop_last=False,
                num_workers=num_workers
            )

            
            for epoch in range(start_epoch, num_epochs):
                train_loss, train_acc = self.train_step(train_loader, 
                                            gradient_accumulation=gradient_accumulation)
                
                self.update_history('train', train_loss, train_acc)
    
                print(f'EPOCH: {epoch+1}/{num_epochs}')
                print(f'Train Loss: {train_loss}')
                print(f'Train Acc: {train_acc}')

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

                    self.update_history('val', val_loss, val_acc)

                    print(f'Validation Loss: {val_loss}')
                    print(f'Validation Acc: {val_acc}')

                
                
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

                self.save_checkpoint(HTER, fold, epoch)

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
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        fold = checkpoint['fold']
        return start_epoch  # Return the starting epoch

    def save_checkpoint(self, hter, fold, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if hter < self.best_val_acc:
            self.best_val_acc = hter
            model_path = os.path.join(self.checkpoint_dir, f'best_model_fold_{fold}.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
                'fold': fold,
            }, model_path)
            print(f'best model at fold {fold} and epoch {epoch} with HTER {hter}')
        else:
            model_path = os.path.join(self.checkpoint_dir, f'fold_{fold}_epoch_{epoch}.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
                'fold': fold,
            }, model_path)
        if self.upload_gcs:
            self.uplpoad_gcs(model_path, f'model/{model_path.split("/")[-1]}')
        ckpt_dir = [os.path.join(self.checkpoint_dir,model_file) for model_file in os.listdir(self.checkpoint_dir) if model_file.startswith('fold_')]
        if len(ckpt_dir) > self.save_total_limit:
            oldest_file = min(ckpt_dir, key=os.path.getctime)
            os.remove(oldest_file)

#########===================================== main ===============================#########

if __name__ == "__main__":
    seed_everything(42)
    train_config = TRAINCONFIG()

    wandb.init(project="Face-Flashing", 
               resume=True, 
               id='training',
               notes="Training ATRFAS model",)

    # load dataset
    dataset = pd.read_csv('train_data.csv')

    model = ATRFAS(num_frames=6, return_proba=True, infer_type='training')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate, weight_decay=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      lambda_param=train_config.lambda_param, 
                      scheduler=scheduler, 
                      device='cpu', 
                      save_total_limit=train_config.save_total_limit,
                      checkpoint_dir='checkpoints'
                      )
    
    # Set num_workers to 0 for debugging
    trainer.fit(dataset=dataset,
                num_epochs=train_config.num_epochs,
                batch_size=train_config.batch_size, 
                num_fold=train_config.num_fold,
                eval_per_epoch=train_config.eval_per_epoch, 
                gradient_accumulation=train_config.gradient_accumulation)  # Set num_workers to 0
    print("Training finished")