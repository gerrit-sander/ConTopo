import argparse
from torchvision import transforms
from torchvision import datasets
from utils import TwoCropTransform, load_cifar10_metadata
import torch
import torch.backends.cudnn as cudnn
from networks.modified_ResNet18 import ProjectionResNet18, LinearResNet18, LinearClassifier
from losses.cosine_contrastive import CosineContrastiveLoss
from losses.topographic import Global_Topographic_Loss, Local_WS_Loss


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate for optimizer')
    parser.add_argument('--embedding_dim', type=int, default=256, help='dimension of the embedding space')
    parser.add_argument('--task_loss', type=str, default='cosine_contrastive', choices=['cosine_contrastive', 'supcon', 'cross_entropy', 'SimCLR'], help='type of loss function to use')
    parser.add_argument('--topographic_loss', type=str, default='global', choices=['global', 'ws'], help='type of topographic loss to use')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training (e.g., cuda or cpu)')
    parser.add_argument('--margin_same', type=float, default=0.3, help='margin for same animacy pairs in cosine contrastive loss')
    parser.add_argument('--margin_diff', type=float, default=0.5, help='margin for different animacy pairs in cosine contrastive loss')
    parser.add_argument('--readout_epochs', type=int, default=100, help='number of epochs for readout training')
    parser.add_argument('--projection_dim', type=int, default=128, help='dimension of the projection head for contrastive learning')
    parser.add_argument('--topographic_loss_lambda', type=float, default=1.0, help='weight for the topographic loss')

    arguments = parser.parse_args()
    return arguments

def cifar10_loader(arguments):

    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=TwoCropTransform(train_transform))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=True,
    )
    return train_loader

def setup_model(arguments):
    if arguments.task_loss == 'cosine_contrastive':
        cifar10_config = load_cifar10_metadata()
        model = ProjectionResNet18(feat_dim=arguments.projection_dim, repr_dim=arguments.embedding_dim)
        task_loss = CosineContrastiveLoss(
            superclass=cifar10_config["ANIMACY"],
            superclass_mapping=cifar10_config["ANIMACY_MAPPING"],
            classnames=cifar10_config["CIFAR10_CLASSES"],
            margin_same=arguments.margin_same,
            margin_diff=arguments.margin_diff
        )
    elif arguments.task_loss == 'supcon':
        model = ProjectionResNet18(feat_dim=arguments.projection_dim, repr_dim=arguments.embedding_dim)
        ### STILL TODO: Implement SupCon loss ###
        raise NotImplementedError("SupCon loss is not implemented yet.")
    elif arguments.task_loss == 'cross_entropy':
        model = LinearResNet18(num_classes=10, repr_dim=arguments.embedding_dim)
        task_loss = torch.nn.CrossEntropyLoss()
    elif arguments.task_loss == 'SimCLR':
        model = ProjectionResNet18(feat_dim=arguments.projection_dim, repr_dim=arguments.embedding_dim)
        ### STILL TODO: Implement SimCLR loss ###
        raise NotImplementedError("SimCLR loss is not implemented yet.")

    if arguments.topographic_loss == 'global':
        topographic_loss = Global_Topographic_Loss(weight=arguments.topographic_loss_lambda, repr_dim=arguments.embedding_dim)
    elif arguments.topographic_loss == 'ws':
        topographic_loss = Local_WS_Loss(weight=arguments.topographic_loss_lambda, repr_dim=arguments.embedding_dim)
    else:
        topographic_loss = None

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, task_loss, topographic_loss