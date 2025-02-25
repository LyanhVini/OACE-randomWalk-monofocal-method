"""
Script principal para executar todo o pipeline
Códigos de Auxílio:
    - https://github.com/rasbt/deeplearning-models/blob/master/pytorch-lightning_ipynb/cnn/cnn-alexnet-grouped-cifar10.ipynb
    - https://github.com/simoninithomas/cifar-10-classifier-pytorch/blob/master/PyTorch%20Cifar-10%20Classifier.ipynb
    - https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
"""
from datasets import *
from models import *
from metrics import *
from hyperparameter_optimization import *
from validation import *
import json
import pickle

def main():
    # Definir o caminho do checkpoint
    
    datasets_options = {
        '1': 'Chest X-Ray',
        '2': 'CIFAR-10',
        '3': 'TrashNet'
    }
    print("Selecione um dataset para rodar o método OACE:")
    print("1: Chest X-Ray")
    print("2: CIFAR-10")
    print("3: TrashNet")
    
    choice = input("\nEscolha o número que corresponda ao dataset: ").strip()
    
    if choice == '1':
        lbd = 0.7
        train_dir = 'datasets/chest_xray/train'
        val_dir = 'datasets/chest_xray/val'
        test_dir = 'datasets/chest_xray/test'
        trainLoader, validLoader, testLoader, classes = chest_x_ray(train_dir, val_dir, test_dir, batch_size=64)
        dataset_name = datasets_options['1']
        subsample_loader, subsample_classes = chest_x_ray_subsample(test_dir)  
    elif choice == '2':
        lbd = 0.5
        trainLoader, validLoader, testLoader, classes = cifar_10(batch_size=64)
        dataset_name = datasets_options['2']
        subsample_loader, subsample_classes = cifar_10_subsample()      
    elif choice == '3':
        lbd = 0.25
        dataset_dir = 'datasets/dataset-resized'
        trainLoader, validLoader, testLoader, classes = trashNet(dataset_dir, batch_size=64)
        dataset_name = datasets_options['3']
        subsample_loader, subsample_classes = trashNet_subsample(dataset_dir) 
    else:
        print("Escolha Inv1álida.")
        return

    models = [
        ("EfficientNetB0", get_efficientnet_b0),
        ("MobileNetV2", get_mobilenet_v2),
        ("ResNet50", get_resnet50),
        ("InceptionV3", get_inception_v3),
        ("VGG16", get_vgg16)
    ]
    # Informações sobre o ambiente de execução
    print("INFORMAÇÕES SOBRE O AMBIENTE DE EXECUÇÃO: ")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    print("\nIniciando o aquecimento dos modelos...\n")
    # Subsample para aquecimento
    print("Classes dos dados para aquecimento:", subsample_classes)
    for images, labels in subsample_loader:
        print("Batchs of images shape:", images.shape)
        print("Batch of labels shape:", labels.shape)
        break

    # Realiza o warm-up dos modelos
    warm_up_metrics = warm_up_models(models, subsample_loader, device)

    print(warm_up_metrics)
    
    with open('warm_up_metrics.pkl', 'wb') as f:
        pickle.dump(warm_up_metrics, f)
    
    # Otimização de hiperparâmetros com o dataset escolhido
    print(f"\nExecutando o método para o dataset: {dataset_name}...\n")

    checkpoint_path = "checkpoint.pkl"
    best_model, best_score, best_solution, metrics, oace_metrics = optimize_hyperparameters(models, 
                                                                                            trainLoader, testLoader, validLoader, 
                                                                                            classes, lbd, wa, wc,
                                                                                            dataset_name, checkpoint_path, 
                                                                                            num_rounds=3, iterations_per_round=5, 
                                                                                            save_checkpoint_every=3)

    print(f"Best Model: {best_model}")
    print(f"Best Score: {best_score}")
    print(f"Best Solution: {best_solution}")
    print(f"Método Finalizado: ")

    with open('ML_metrics.json', 'w') as f:
        json.dump(metrics, f)       
    with open('OACE_metrics.json', 'w') as f:
        json.dump(oace_metrics, f)

    #Salvando melhor modelo
    torch.save(best_model, "full_best_model.pt")

    print(f"\nMétodo finalizado para o dataset: {dataset_name}")

if __name__ == "__main__": 
    main()