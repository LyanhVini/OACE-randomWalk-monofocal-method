"""
Definição e processamento dos datasets.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

### SAMPLE DATASETS FOR TRAINING:

def chest_x_ray(train_dir, val_dir, test_dir, batch_size=32, num_workers=4):  
    """Duas classes"""
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])    
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transform['train']),
        'val': datasets.ImageFolder(val_dir, transform=data_transform['val']),
        'test': datasets.ImageFolder(test_dir, transform=data_transform['test'])
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=True, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=1, shuffle=True, num_workers=num_workers)
    }

    classes = image_datasets['train'].classes
    return dataloaders['train'], dataloaders['val'], dataloaders['test'], classes

def cifar_10(n_valid=0.2, batch_size=64, num_workers=4):
    """10 Classes"""
    
    transform_train = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  
    ])
    
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

    n_train = len(train_data)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    split = int(np.floor(n_valid * n_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Prepare data loaders (combine dataset and sampler)
    trainLoader = torch.utils.data.DataLoader(train_data,
                                            batch_size = batch_size,
                                            sampler = train_sampler,
                                            num_workers = num_workers)

    validLoader = torch.utils.data.DataLoader(train_data,
                                            batch_size = batch_size,
                                            sampler = valid_sampler,
                                            num_workers = num_workers)

    testLoader = torch.utils.data.DataLoader(test_data,
                                            batch_size = batch_size,
                                            num_workers = num_workers)

    classes = train_data.classes
    return trainLoader, validLoader, testLoader, classes

def trashNet(dataset_dir, batch_size=32, val_split=0.2, test_split=0.1, num_workers=4):
    """6 Classes"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }
    
    dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms['val_test'])
        
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataset.dataset.transform = data_transforms['train']
        
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }

    classes = dataset.classes
    return dataloaders['train'], dataloaders['val'], dataloaders['test'], classes

def verify_dataset(trainloader, valloader, testloader):
    dataset_sizes = {
        'train': len(trainloader.dataset),
        'val': len(valloader.dataset),
        'test': len(testloader.dataset)
    }
    
    # Verificar se as classes estão diretamente acessíveis ou dentro de Subset
    if hasattr(trainloader.dataset, 'classes'):
        classes = trainloader.dataset.classes
    elif hasattr(trainloader.dataset.dataset, 'classes'):
        classes = trainloader.dataset.dataset.classes
    else:
        raise AttributeError("Classes não encontradas no dataset")

    print(f"Tamanhos do dataset: {dataset_sizes}")
    print(f"Classes: {classes}")
    
###  SUBSAMPLES DATASETS FOR WARM:
    
def chest_x_ray_subsample(test_dir, num_workers=4, class_labels=['PNEUMONIA', 'NORMAL']):
    """Chest X-Ray with only two selected classes for warm"""
    data_transform = {
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    
    dataset_chest_x_ray = datasets.ImageFolder(test_dir, transform=data_transform)
    dataLoader = DataLoader(dataset_chest_x_ray, batch_size=64, shuffle=True, num_workers=num_workers)
    classes = class_labels
    
    return dataLoader, classes

def cifar_10_subsample(batch_size=64, num_workers=4, class_labels=['cat', 'dog']):
    """CIFAR-10 with only two selected classes for warm"""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # Load the entire CIFAR-10 test dataset
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)

    # Filter out only the specified classes
    test_indices = [i for i, label in enumerate(test_data.targets) if test_data.classes[label] in class_labels]

    # Map the selected classes to new labels (0 and 1)
    test_data.targets = [test_data.targets[i] for i in test_indices]
    test_data.data = test_data.data[test_indices]

    label_map = {test_data.classes.index(class_labels[0]): 0,
                 test_data.classes.index(class_labels[1]): 1}

    test_data.targets = [label_map[label] for label in test_data.targets]

    # Prepare data loader for test data
    testLoader = DataLoader(test_data, 
                            batch_size=batch_size,
                            num_workers=num_workers)
    
    classes = class_labels
    
    return testLoader, classes

def trashNet_subsample(dataset_dir, batch_size=64, num_workers=4, class_labels=['glass', 'plastic']):
    """TrashNet with only two selected classes for warm"""
    data_transforms = {
        'val_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms['val_test'])
    
    # Filter out only the specified classes
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(dataset.classes)}
    selected_class_idxs = [class_to_idx[cls_name] for cls_name in class_labels]
    
    indices = [i for i, (path, label) in enumerate(dataset.samples) if label in selected_class_idxs]
    dataset.samples = [dataset.samples[i] for i in indices]
    dataset.targets = [dataset.targets[i] for i in indices]
    
    # Map the selected classes to new labels (0 and 1)
    label_map = {selected_class_idxs[0]: 0, selected_class_idxs[1]: 1}
    dataset.targets = [label_map[label] for label in dataset.targets]

    # Prepare data loader for test data
    testLoader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    
    classes = class_labels
    return testLoader, classes