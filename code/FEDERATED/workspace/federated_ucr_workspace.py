import os
import glob
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from openfl.interface.interactive_api.federation import Federation
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment
from copy import deepcopy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import tqdm
import wandb

LOG_WANDB = True

myseed = int(sys.argv[1])
mydataset = sys.argv[2]

torch.manual_seed(myseed)
np.random.seed(myseed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create a federation

# please use the same identificator that was used in signed certificate
client_id = 'api'
cert_dir = 'cert'
director_node_fqdn = 'worker-1'
# 1) Run with API layer - Director mTLS 
# If the user wants to enable mTLS their must provide CA root chain, and signed key pair to the federation interface
# cert_chain = f'{cert_dir}/root_ca.crt'
# api_certificate = f'{cert_dir}/{client_id}.crt'
# api_private_key = f'{cert_dir}/{client_id}.key'

# federation = Federation(client_id=client_id, director_node_fqdn=director_node_fqdn, director_port='50051',
#                        cert_chain=cert_chain, api_cert=api_certificate, api_private_key=api_private_key)

# --------------------------------------------------------------------------------------------------------------------

# 2) Run with TLS disabled (trusted environment)
# Federation can also determine local fqdn automatically
federation = Federation(client_id=client_id, director_node_fqdn=director_node_fqdn, director_port='50051', tls=False)

federation.target_shape

shard_registry = federation.get_shard_registry()
shard_registry

# First, request a dummy_shard_desc that holds information about the federated dataset 
dummy_shard_desc = federation.get_dummy_shard_descriptor(size=10)
dummy_shard_dataset = dummy_shard_desc.get_dataset('train')
sample, target = dummy_shard_dataset[0]
print(sample.shape)
print(target.shape)

class TransformedDataset(Dataset):
    """Image UCR Dataset."""

    def __init__(self, dataset, transform=None, target_transform=None):
        """Initialize Dataset."""
        self.dataset = dataset
        #self.transform = transform
        #self.target_transform = target_transform

    def __len__(self):
        """Length of dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        feature, label = self.dataset[index]
        #label = self.target_transform(label) if self.target_transform else label
        #feature = self.transform(feature) if self.transform else feature
        return feature, label
    
class UCRDataset(DataInterface):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    @property
    def shard_descriptor(self):
        return self._shard_descriptor
        
    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor
        
        self.train_set = TransformedDataset(
            self._shard_descriptor.get_dataset('train'),
            #transform=training_transform
        )
        self.valid_set = TransformedDataset(
            self._shard_descriptor.get_dataset('val'),
            #transform=valid_transform
        )

    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        generator=torch.Generator()
        generator.manual_seed(myseed)
        return DataLoader(
            self.train_set, batch_size=self.kwargs['train_bs'], shuffle=True, generator=generator, drop_last=True
            )

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        return DataLoader(self.valid_set, batch_size=self.kwargs['valid_bs'], drop_last=True)

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_set)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.valid_set)  
    

if int(federation.target_shape[0]) > 33:
    fed_dataset = UCRDataset(train_bs=8, valid_bs=8)
elif 8 < int(federation.target_shape[0]) < 33:
    fed_dataset = UCRDataset(train_bs=4, valid_bs=4)
else:
    fed_dataset = UCRDataset(train_bs=2, valid_bs=2)

in_channels = sample.shape[1]
num_classes = target.shape[1]

class MLP(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLP, self).__init__()
        torch.manual_seed(myseed)
        self.ln1 = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        #out = self.dropout1(x)
        out = self.ln1(x)

        return out
                
if num_classes == 2:
    model_net = MLP(in_channels, num_classes-1)
else:
    model_net = MLP(in_channels, num_classes)
print(model_net)

def init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0)
        nn.init.constant_(layer.bias.data, 0)
        
params_to_update = []
for param in model_net.parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
    
model_net.apply(init)

# Define an optimizer
optimizer = optim.Adam(params_to_update, lr=0.001)
# Define a loss
if num_classes == 2:
    criterion = nn.BCEWithLogitsLoss() 
else:
    criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, min_lr = 1e-8)

def cross_entropy(output, target):
    """Cross-entropy metric
    """
    loss = criterion(output, target)
    return loss

framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
model_interface = ModelInterface(model=model_net, optimizer=optimizer, framework_plugin=framework_adapter)

# Save the initial model state
initial_model = deepcopy(model_net)

task_interface = TaskInterface()

train_custom_params={'f1_score': f1_score}

# The Interactive API supports registering functions definied in main module or imported.
def function_defined_in_notebook(some_parameter):
    print(f'Also I accept a parameter and it is {some_parameter}')

# Task interface currently supports only standalone functions.
@task_interface.add_kwargs(**train_custom_params)
@task_interface.add_kwargs(**{'some_parameter': 42})
@task_interface.register_fl_task(model='net_model', data_loader='train_loader', \
                     device='device', optimizer='optimizer') 


def train(net_model, train_loader, optimizer, device, f1_score, loss_fn=cross_entropy, some_parameter=None):
    torch.manual_seed(myseed)
    #fedcurv.on_train_begin(net_model)
    device='cuda'
    function_defined_in_notebook(some_parameter)
    
    train_loader = tqdm.tqdm(train_loader, desc="train")
    net_model.train()
    net_model.to(device)

    losses = []
    total_acc, sum_f1 = 0, 0
    total_samples = 0
    epochs = 1
    
    for epoch in range(epochs):
        for data, target in train_loader:
            samples = sample.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), torch.tensor(
                target).to(device, dtype=torch.int64)
            optimizer.zero_grad()
            #output = torch.flatten(net_model(data))
            output = net_model(data)
            if num_classes == 2:
                output = output.float()
                target = target.unsqueeze(1)
                target = target.float()
            loss = loss_fn(output=output, target=target)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy()) 
            
            #accuracy
            if num_classes == 2:
                pred_labels = (output >= 0).float() # Binarize predictions to 0 and 1
                sum_f1 += f1_score(target.cpu().numpy(), pred_labels.cpu().numpy(), average='binary', zero_division=0)
            else:
                _,pred_labels = torch.max(output, dim=1)
                sum_f1 += f1_score(target.cpu().numpy(), pred_labels.cpu().numpy(), average='macro', zero_division=0)
            
            batch_accuracy = (pred_labels == target).sum().item()/data.size(0)            
            total_acc += batch_accuracy  
            
            
        if LOG_WANDB:
            wandb.run.summary["step"] = epoch
            wandb.log({'Training loss': np.mean(losses), 'Training accuracy': total_acc/len(train_loader)})

    return {'Train_Loss': np.mean(losses),
           'Train_Accuracy': total_acc/len(train_loader),
           'Train_f1': sum_f1/len(train_loader)}

val_custom_params={'f1_score': f1_score}
@task_interface.register_fl_task(model='net_model', data_loader='val_loader', device='device')    
@task_interface.add_kwargs(**val_custom_params)
def validate(net_model, val_loader, device, f1_score):
    torch.manual_seed(myseed)
    device = torch.device('cuda')
    net_model.eval()
    net_model.to(device)
    losses = []
    total_acc, sum_f1 = 0, 0
    
    val_loader = tqdm.tqdm(val_loader, desc="validate")
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), \
                torch.tensor(target).to(device, dtype=torch.int64)
            #output = torch.flatten(net_model(data))
            output = net_model(data)
            if num_classes == 2:
                output = output.float()
                target = target.unsqueeze(1)
                target = target.float()
            loss = criterion(output, target)
            losses.append(loss.detach().cpu().numpy())
            
            #accuracy
            if num_classes == 2:
                pred_labels = (output >= 0).float() # Binarize predictions to 0 and 1
                sum_f1 += f1_score(target.cpu().numpy(), pred_labels.cpu().numpy(), average='binary', zero_division=0)
            else:
                _,pred_labels = torch.max(output, dim=1)
                sum_f1 += f1_score(target.cpu().numpy(), pred_labels.cpu().numpy(), average='macro', zero_division=0)

            batch_accuracy = (pred_labels == target).sum().item()/data.size(0)
            total_acc += batch_accuracy  
            
        if LOG_WANDB:
            wandb.log({'Test loss': np.mean(losses), 'Test accuracy': total_acc/len(val_loader)})
            
    return {'Test_loss': np.mean(losses),
            'Test_accuracy': total_acc/len(val_loader),
            'Test_f1': sum_f1/len(val_loader)}


# create an experimnet in federation
experiment_name = f"UCR_{mydataset}_seed{myseed}"
fl_experiment = FLExperiment(federation=federation, experiment_name=experiment_name)

# The following command zips the workspace and python requirements to be transfered to collaborator nodes
fl_experiment.start(
    model_provider=model_interface, 
    task_keeper=task_interface,
    data_loader=fed_dataset,
    rounds_to_train=100,
    opt_treatment='CONTINUE_GLOBAL'
)

# If user want to stop IPython session, then reconnect and check how experiment is going
# fl_experiment.restore_experiment_state(model_interface)

fl_experiment.stream_metrics(tensorboard_logs=True)