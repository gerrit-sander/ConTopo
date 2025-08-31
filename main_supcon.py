import argparse
from argparse import BooleanOptionalAction
from torchvision import transforms
from torchvision import datasets
from utils.train import AverageMeter, save_checkpoint, unwrap, tb_logger, TwoCropTransform, grad_norm
import torch
import torch.backends.cudnn as cudnn
from networks.shallowCNN import ProjectionShallowCNN, LinearClassifier
from networks.modified_ResNet18 import ProjectionResNet18
from losses.topographic import Global_Topographic_Loss, Local_WS_Loss
from losses.supcon import SupConLoss
import time
import os
import sys
import torch.optim as optim


def parse_arguments():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--trial', type=int, default=0, help='trial number for multiple runs (used in naming folders)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')
    parser.add_argument('--task_method', type=str, default='supcon', choices=['supcon', 'simclr'], help='type of task loss to use')

    # Topographic Loss settings
    parser.add_argument('topography_type', type=str, choices=['global', 'ws'], help='type of topographic loss to use')
    parser.add_argument('--topographic_loss_rho', type=float, default=0.05, help='balancing factor of the two losses')

    # Optimization settings
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--readout_epochs', type=int, default=20, help='number of epochs for readout training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

    # Model settings
    parser.add_argument('model_type', type=str, choices=['shallowcnn', 'resnet18'], help='type of model to use')
    parser.add_argument('--embedding_dim', type=int, default=256, help='dimension of the embedding space')
    parser.add_argument('--projection_dim', type=int, default=128, help='dimension of the projection head for contrastive learning')
    parser.add_argument('--use_dropout', action=BooleanOptionalAction, default=True, help='use dropout')
    parser.add_argument('--p_dropout', type=float, default=0.5, help='dropout probability (if applicable)')

    arguments = parser.parse_args()

    subdir = 'ResNet18' if arguments.model_type == 'resnet18' else 'ShallowCNN'
    arguments.model_folder = f'./save/{subdir}/models'
    arguments.tensorboard_folder = f'./save/{subdir}/tensorboard/{arguments.task_method}'
    arguments.dataset_folder = './dataset'
    arguments.save_freq = max(1, arguments.epochs // 10)  # Save every 10% of epochs, rounded up

    arguments.model_name = '{}_{}topo_{}embdims_{}projdims_{}rho_{}epochs_{}bsz_nwork{}_readep{}_lr{}_{}dropout'.format(
        arguments.task_method,
        arguments.topography_type,
        arguments.embedding_dim, 
        arguments.projection_dim, 
        arguments.topographic_loss_rho, 
        arguments.epochs,
        arguments.batch_size,
        arguments.num_workers,
        arguments.readout_epochs,
        arguments.learning_rate,
        arguments.p_dropout if arguments.use_dropout else 0.0,
    )

    run_name = f"trial_{arguments.trial:02d}"

    arguments.tensorboard_folder = os.path.join(arguments.tensorboard_folder, arguments.model_name, run_name)
    os.makedirs(arguments.tensorboard_folder, exist_ok=True)

    arguments.model_folder = os.path.join(arguments.model_folder, arguments.model_name, run_name)
    os.makedirs(arguments.model_folder, exist_ok=True)

    return arguments

def cifar10_loader(arguments):
    # Use standard normalization and data augmentation for CIFAR-10
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Training augmentations (two views for contrastive)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # contrastive val transform for loss validation during training
    val_transform_contrastive = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Datasets
    train_dataset = datasets.CIFAR10(root=arguments.dataset_folder, transform=TwoCropTransform(train_transform), download=True)
    val_dataset = datasets.CIFAR10(root=arguments.dataset_folder, train=False, download=True, transform=val_transform)
    val_dataset_contrastive = datasets.CIFAR10(root=arguments.dataset_folder, train=False, download=True, transform=TwoCropTransform(val_transform_contrastive))

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=arguments.batch_size, shuffle=True,
        num_workers=arguments.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=arguments.batch_size, shuffle=False,
        num_workers=arguments.num_workers, pin_memory=True,
    )
    val_contrastive_loader = torch.utils.data.DataLoader(
        val_dataset_contrastive, batch_size=arguments.batch_size, shuffle=False,
        num_workers=arguments.num_workers, pin_memory=True,
    )

    return train_loader, val_loader, val_contrastive_loader

def setup_model(arguments):

    # Select the proper model
    if arguments.model_type == 'shallowcnn':
        model = ProjectionShallowCNN(emb_dim=arguments.embedding_dim, feat_dim=arguments.projection_dim, ret_emb=True, use_dropout=arguments.use_dropout, p_dropout=arguments.p_dropout)
    elif arguments.model_type == 'resnet18':
        model = ProjectionResNet18(emb_dim=arguments.embedding_dim, feat_dim=arguments.projection_dim, ret_emb=True, use_dropout=arguments.use_dropout, p_dropout=arguments.p_dropout)

    # Select the task loss (SupConLoss implements both SupCon loss and SimCLR loss)
    task_loss = SupConLoss(temperature=0.07)

    # Select the topographic loss type
    if arguments.topography_type == 'global':
        topographic_loss = Global_Topographic_Loss(weight=1.0, emb_dim=arguments.embedding_dim)
    elif arguments.topography_type == 'ws':
        topographic_loss = Local_WS_Loss(weight=1.0)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        if isinstance(task_loss, torch.nn.Module):
            task_loss = task_loss.cuda()
        if isinstance(topographic_loss, torch.nn.Module):
            topographic_loss = topographic_loss.cuda()
        cudnn.benchmark = True

    return model, task_loss, topographic_loss

def train(train_loader, model, task_loss, topographic_loss, optimizer, epoch, arguments):
    model.train()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    topographic_losses = AverageMeter()
    task_losses = AverageMeter()
    data_time = AverageMeter()
    lambda_hat_meter = AverageMeter()

    # Dynamic loss balancing:
    # - 0 < rho < 1  → task loss dominates
    # - rho = 1      → equal weight
    # - rho > 1      → topographic loss dominates
    # eps avoids div-by-zero; beta is EMA smoothing for lambda_hat
    rho = arguments.topographic_loss_rho     
    beta = 0.1
    eps = 1e-8
    lambda_max = 1e4
    lambda_hat = None  # smoothed scale for topo loss

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # merge the two augmented views along batch dim → (2*B, C, H, W)
        images = torch.cat([images[0], images[1]], dim=0)

        device = next(model.parameters()).device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bsz = labels.shape[0]

        # forward through encoder + projection head
        embeddings, features = model(images)

        # reshape features to (B, 2, feat_dim) for SupCon/SimCLR
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # task loss: supervised contrastive or SimCLR variant
        if arguments.task_method == 'supcon':
            task_loss_value = task_loss(features, labels)
        elif arguments.task_method == 'simclr':
            task_loss_value = task_loss(features)
        else:
            raise ValueError(f"Unknown task method: {arguments.task_method}")

        # topographic term: local WS acts on final linear layer; global acts on embeddings
        if arguments.topography_type == 'ws':
            base = unwrap(model)
            linear_layer = base.encoder.fc
            topographic_loss_value = topographic_loss(linear_layer=linear_layer)
            measure_params = list(linear_layer.parameters())
        elif arguments.topography_type == 'global':
            topographic_loss_value = topographic_loss(embeddings)
            if isinstance(model, torch.nn.DataParallel):
                measure_params = [p for p in model.encoder.module.parameters() if p.requires_grad]
            else:
                measure_params = [p for p in model.encoder.parameters() if p.requires_grad]
        else:
            measure_params = [p for p in model.parameters() if p.requires_grad]  # fallback

        # gradient-norm matching to set lambda
        nt = grad_norm(task_loss_value, measure_params)
        np_ = grad_norm(topographic_loss_value, measure_params)
        target_lambda = (rho * nt / (np_ + eps)).clamp(0.0, lambda_max).detach()
        if lambda_hat is None:
            lambda_hat = target_lambda
        else:
            lambda_hat = (1 - beta) * lambda_hat + beta * target_lambda  # EMA smoothing
        # Track lambda_hat value
        lambda_hat_meter.update(float(lambda_hat.detach().cpu()), bsz)

        # combined objective
        loss = task_loss_value + lambda_hat.detach() * topographic_loss_value

        # update meters
        losses.update(loss.item(), bsz)
        task_losses.update(task_loss_value.item(), bsz)
        topographic_losses.update(topographic_loss_value.item(), bsz)

        # standard optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        # periodic log (includes current smoothed lambda)
        if (idx + 1) % arguments.print_freq == 0:
            lam_val = float(lambda_hat.detach().cpu()) if lambda_hat is not None else 1.0
            print(f'Epoch: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Topographic Loss {topographic_losses.val:.4f} ({topographic_losses.avg:.4f})\t'
                  f'Task Loss {task_losses.val:.4f} ({task_losses.avg:.4f})\t'
                  f'Lambda {lam_val:.4f}\t'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})')
            sys.stdout.flush()

    return losses.avg, topographic_losses.avg, task_losses.avg, lambda_hat_meter.avg

def validate(val_loader, model, criterion, arguments):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():  # eval without grads
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            device = next(model.parameters()).device
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bsz = labels.shape[0]

            # forward and compute loss
            output = model(images)
            loss = criterion(output, labels)

            # update loss/accuracy meters
            losses.update(loss.item(), bsz)
            _, predicted = output.max(1)
            correct = predicted.eq(labels).sum().item()
            acc_value = correct / bsz
            acc.update(acc_value, bsz)

            # time bookkeeping
            batch_time.update(time.time() - end)
            end = time.time()

            # periodic validation log
            if idx % arguments.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, acc=acc))

    print(' * Acc@1 {acc.avg:.3f}'.format(acc=acc))
    return losses.avg, acc.avg

def validate_contrastive(val_loader, model, task_loss, topographic_loss, arguments):
    """Eval SupCon/SimCLR + topo on the validation set with TWO views (no grads)."""
    model.eval()
    task_losses = AverageMeter()
    topo_losses = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            device = next(model.parameters()).device
            if isinstance(images, (list, tuple)):
                v1, v2 = images
                B = labels.size(0)
                x = torch.cat([v1, v2], dim=0)
            else:
                B = labels.size(0)
                x = torch.cat([images, images], dim=0)

            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            embeddings, features = model(x)

            f1, f2 = torch.split(features, [B, B], dim=0)
            feats_2view = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            # task loss
            if arguments.task_method == 'supcon':
                task_val = task_loss(feats_2view, labels)
            elif arguments.task_method == 'simclr':
                task_val = task_loss(feats_2view)
            else:
                raise ValueError(f"Unknown task method: {arguments.task_method}")

            if arguments.topography_type == 'ws':
                base = unwrap(model)
                topo_val = topographic_loss(linear_layer=base.encoder.fc)
            else:  # 'global'
                topo_val = topographic_loss(embeddings)

            task_losses.update(task_val.item(), B)
            topo_losses.update(topo_val.item(), B)

    return task_losses.avg, topo_losses.avg

def main():
    arguments = parse_arguments()

    train_loader, val_loader, val_contrastive_loader = cifar10_loader(arguments)

    ### CONTRASTIVE LEARNING ###
    model, task_loss, topographic_loss = setup_model(arguments)

    optimizer = optim.SGD(model.parameters(), lr=arguments.learning_rate, momentum=0.9, weight_decay=1e-4)
    
    logger = tb_logger.Logger(logdir=arguments.tensorboard_folder, flush_secs=2)

    best_contrastive_loss = float('inf')

    for epoch in range(1, arguments.epochs + 1):

        time1 = time.time()
        avg_loss, avg_topoloss, avg_taskloss, avg_lambda_hat = train(
            train_loader, model, task_loss, topographic_loss, optimizer, epoch, arguments
        )
        val_task_loss, val_topo_loss = validate_contrastive(
            val_contrastive_loader, model, task_loss, topographic_loss, arguments
        )

        val_total_loss = val_task_loss + avg_lambda_hat * val_topo_loss

        logger.log_value('val_task_loss', val_task_loss, epoch)
        logger.log_value('val_topographic_loss', val_topo_loss, epoch)
        logger.log_value('val_total_loss', val_total_loss, epoch)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('total_loss', avg_loss, epoch)
        logger.log_value('task_loss', avg_taskloss, epoch)
        logger.log_value('topographic_loss', avg_topoloss, epoch)
        logger.log_value('lambda_hat', avg_lambda_hat, epoch)

        # ------ CONTRASTIVE CHECKPOINTS ------
        # Periodic full snapshot (encoder + optimizer + metrics)
        if (epoch % arguments.save_freq) == 0:
            state = {
                'stage': 'contrastive',
                'epoch': epoch,
                'state_dict': unwrap(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(arguments),
                'metrics': {
                    'total_loss': avg_loss,
                    'task_loss': avg_taskloss,
                    'topographic_loss': avg_topoloss,
                }
            }
            ckpt_path = os.path.join(arguments.model_folder, f'contrastive_epoch{epoch:04d}.pth')
            save_checkpoint(ckpt_path, state)

        # Track best model by lowest total loss
        if avg_loss < best_contrastive_loss:
            best_contrastive_loss = avg_loss
            best_state = {
                'stage': 'contrastive',
                'epoch': epoch,
                'state_dict': unwrap(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(arguments),
                'metrics': {
                    'total_loss': avg_loss,
                    'task_loss': avg_taskloss,
                    'topographic_loss': avg_topoloss,
                }
            }
            best_path = os.path.join(arguments.model_folder, 'contrastive_best.pth')
            save_checkpoint(best_path, best_state)

    # Always save the last contrastive snapshot (for resuming)
    final_contrastive = {
        'stage': 'contrastive',
        'epoch': arguments.epochs,
        'state_dict': unwrap(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(arguments),
    }
    save_checkpoint(os.path.join(arguments.model_folder, 'contrastive_last.pth'), final_contrastive)

    ### LINEAR READOUT TRAINING ###
    # Freeze the pretrained encoder; keep it in eval for stable features
    for p in model.parameters():
        p.requires_grad = False
    unwrap(model).eval()  # safely puts underlying module in eval, even if DataParallel

    linear_clf = LinearClassifier(emb_dim=arguments.embedding_dim, num_classes=10)
    if torch.cuda.is_available():
        linear_clf = linear_clf.cuda()

    class EncoderWithLinear(torch.nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
        def forward(self, x):
            # cache-free forward: encoder frozen, grads disabled
            with torch.no_grad():
                embeddings, _ = self.encoder(x)
            logits = self.classifier(embeddings)
            return logits

    readout_model = EncoderWithLinear(model, linear_clf)

    readout_optimizer = optim.SGD(linear_clf.parameters(),
                                  lr=arguments.learning_rate,
                                  momentum=0.9,
                                  weight_decay=0.0)

    criterion_ce = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion_ce = criterion_ce.cuda()

    best_val_acc = 0.0
    last_val_acc = 0.0

    for epoch in range(1, arguments.readout_epochs + 1):
        readout_model.train()
        unwrap(model).eval()  # keep encoder frozen/eval

        train_loss_meter = AverageMeter()
        end = time.time()

        for idx, (images, labels) in enumerate(train_loader):
            
            if isinstance(images, (list, tuple)):
                images = images[0]     # use one view for linear probe
            device = next(linear_clf.parameters()).device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = readout_model(images)
            loss = criterion_ce(logits, labels)

            readout_optimizer.zero_grad()
            loss.backward()
            readout_optimizer.step()

            bsz = labels.size(0)
            train_loss_meter.update(loss.item(), bsz)

            # periodic readout training log
            if (idx + 1) % arguments.print_freq == 0:
                print(f'[Linear] Epoch: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                      f'Loss {train_loss_meter.val:.4f} ({train_loss_meter.avg:.4f})')
                sys.stdout.flush()

        print(f'[Linear] epoch {epoch}, train loss {train_loss_meter.avg:.4f}')
        logger.log_value('linear_readout_train_loss', train_loss_meter.avg, epoch)

        # Validation of linear head on frozen features
        val_loss, val_acc = validate(val_loader, readout_model, criterion_ce, arguments)
        last_val_acc = val_acc
        logger.log_value('linear_readout_val_loss', val_loss, epoch)
        logger.log_value('linear_readout_val_acc', val_acc, epoch)

        # Periodic checkpoint of the linear head only (encoder already saved above)
        if (epoch % arguments.save_freq) == 0:
            readout_state = {
                'stage': 'linear_readout',
                'epoch': epoch,
                'linear_state_dict': linear_clf.state_dict(),
                'args': vars(arguments),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            save_checkpoint(
                os.path.join(arguments.model_folder, f'readout_epoch{epoch:04d}.pth'),
                readout_state
            )

        # Keep the best linear head by highest validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_readout_state = {
                'stage': 'linear_readout',
                'epoch': epoch,
                'linear_state_dict': linear_clf.state_dict(),
                'args': vars(arguments),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            save_checkpoint(
                os.path.join(arguments.model_folder, 'readout_best.pth'),
                best_readout_state
            )

    # Always save the final linear head snapshot
    save_checkpoint(
        os.path.join(arguments.model_folder, 'readout_last.pth'),
        {
            'stage': 'linear_readout',
            'epoch': arguments.readout_epochs,
            'linear_state_dict': linear_clf.state_dict(),
            'args': vars(arguments),
            'val_acc': last_val_acc,
        }
    )

    if hasattr(logger, "close"):
        logger.close()
                
if __name__ == '__main__':
    main()