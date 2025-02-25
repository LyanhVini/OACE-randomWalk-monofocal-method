"""
Definição e treinamento dos modelos.
"""
import torch.nn as nn
import torchvision.models as models

def get_efficientnet_b0(num_classes=10):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if hasattr(model.classifier, 'in_features'):
        in_features = model.classifier.in_features
    elif hasattr(model.classifier, 'fc'):
        in_features = model.classifier.fc.in_features
    else:
        in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    #print(model)
    return model

def get_mobilenet_v2(num_classes=10):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    #print(model)
    return model

def get_resnet50(num_classes=10):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #print(model)
    return model

def get_inception_v3(num_classes=10):
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #print(model)
    return model

def get_vgg16(num_classes=10):
    #model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    #model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    #model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    # Congelar as camadas convolucionais
    #for param in model.features.parameters():
    #    param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
"""
def get_vgg19(num_classes=10):
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    # Congelar as camadas convolucionais
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    #print(model)
    return model
"""