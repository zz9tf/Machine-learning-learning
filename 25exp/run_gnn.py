import os
import numpy as np
import torch
import pandas as pd
from gnn import GraphDataset, GNN, DataLoader
from gnn_autoencoder import GNNAutoencoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
from torch.utils.data import random_split
import csv
import matplotlib.pyplot as plt
import argparse
import datetime
import yaml

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UserWarning)

def evaluate_model(model, data_loader, device, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).cpu().numpy()
            # pred = out['graph_logits'].argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            
            # 如果提供了损失函数，计算损失
            if criterion is not None:
                loss = criterion(out, data.y)
                # loss = model.loss_function(out, data.y)['loss']
                total_loss += loss.item() * data.num_graphs
            
            all_preds.extend(pred)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Handle potential warnings using zero_division=0
    try:
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        precision = recall = f1 = 0.0
    
    # Generate confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
    except:
        cm = None
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader.dataset) if criterion is not None else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'loss': avg_loss
    }

# 监控梯度的函数
def log_gradients(model, gradient_stats):
    """记录模型各层的梯度信息"""
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # 计算梯度的统计信息
            grad_abs = param.grad.abs()
            grad_mean = grad_abs.mean().item()
            grad_max = grad_abs.max().item()
            grad_min = grad_abs.min().item()
            grad_std = grad_abs.std().item()
            
            # 如果是第一次记录，初始化字典
            if name not in gradient_stats:
                gradient_stats[name] = {
                    'mean': [], 'max': [], 'min': [], 'std': []
                }
            
            # 记录统计信息
            gradient_stats[name]['mean'].append(grad_mean)
            gradient_stats[name]['max'].append(grad_max)
            gradient_stats[name]['min'].append(grad_min)
            gradient_stats[name]['std'].append(grad_std)
    
    return gradient_stats

def print_gradient_stats(gradient_stats, epoch):
    """打印当前梯度的统计信息"""
    print(f"\n梯度统计信息 (Epoch {epoch}):")
    print("层名称                  平均梯度    最大梯度    最小梯度    标准差")
    print("-" * 80)
    
    # 按梯度大小排序
    # sorted_layers = sorted(
    #     [(name, stats['mean'][-1]) for name, stats in gradient_stats.items()],
    #     key=lambda x: x[1]
    # )
    
    layers = [(name, stats['mean'][-1]) for name, stats in gradient_stats.items()]
    
    for name, _ in layers:
        stats = gradient_stats[name]
        print(f"{name:25s} {stats['mean'][-1]:.8f}  {stats['max'][-1]:.8f}  {stats['min'][-1]:.8f}  {stats['std'][-1]:.8f}")

# 添加一个函数来生成模型结构摘要，类似于TensorFlow的summary
def get_model_summary(model, input_size=None):
    """
    生成类似TensorFlow的模型结构摘要
    """
    result = []
    result.append("Model Structure Summary:")
    result.append("=" * 80)
    result.append("{:<30} {:<25} {:<15}".format("Layer (type)", "Output Shape", "Param #"))
    result.append("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    # 获取所有直接子模块
    for name, layer in model.named_children():
        # 计算该层的参数数量
        layer_params = sum(p.numel() for p in layer.parameters())
        total_params += layer_params
        trainable_params += sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        # 尝试获取输出形状，可能无法确定
        output_shape = "Unknown"
        if hasattr(layer, "out_channels"):
            if hasattr(layer, "in_channels"):
                output_shape = f"[?, {layer.out_channels}]"
        
        result.append("{:<30} {:<25} {:<15,}".format(f"{name} ({layer.__class__.__name__})", 
                                                 output_shape, 
                                                 layer_params))
        
        # 如果该层是嵌套的，递归获取其子层
        if list(layer.children()):
            for subname, sublayer in layer.named_children():
                subparams = sum(p.numel() for p in sublayer.parameters())
                sub_output_shape = "Unknown"
                if hasattr(sublayer, "out_channels"):
                    if hasattr(sublayer, "in_channels"):
                        sub_output_shape = f"[?, {sublayer.out_channels}]"
                
                result.append("{:<30} {:<25} {:<15,}".format(f"  └─{subname} ({sublayer.__class__.__name__})", 
                                                       sub_output_shape, 
                                                       subparams))
    
    result.append("=" * 80)
    result.append(f"Total params: {total_params:,}")
    result.append(f"Trainable params: {trainable_params:,}")
    result.append(f"Non-trainable params: {total_params - trainable_params:,}")
    result.append("=" * 80)
    
    return "\n".join(result)

def main(config_path='config.yaml', plots_dir=None):
    print("Starting GNN training and prediction...")
    
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 提取配置参数
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    output_config = config['output']
    
    # 如果命令行指定了plots_dir，则优先使用
    if plots_dir:
        output_config['plots_dir'] = plots_dir
        
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.csv')
    
    print(f"Loading datasets from {data_dir}")
    
    # Load datasets
    train_dataset = GraphDataset(root=data_dir, mode='train')
    test_dataset = GraphDataset(root=data_dir, mode='test')
    
    # 使用配置中的参数进行训练/验证集划分
    train_size = int(data_config['train_val_split'] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    print(f"Total training dataset size: {len(train_dataset)} graphs")
    print(f"Training subset size: {len(train_subset)} graphs")
    print(f"Validation subset size: {len(val_subset)} graphs")
    print(f"Test dataset size: {len(test_dataset)} graphs")
    
    # 从数据集获取实际的节点标签数量，并更新配置
    model_config['num_node_labels'] = train_dataset.num_node_labels
    print(f"Actual number of node labels: {model_config['num_node_labels']}")
    
    # Create data loaders with batch size from config
    batch_size = training_config['batch_size']
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine number of classes
    num_classes = max(train_dataset.labels_map.values()) + 1
    print(f"Number of classes: {num_classes}")
    
    # Initialize model with parameters from config
    model = GNN(
        hidden_channels=model_config['hidden_channels'],
        out_channels=num_classes,
        num_node_labels=model_config['num_node_labels'],
        embedding_dim=model_config['embedding_dim']
    ).to(device)
    # model = GNNAutoencoder(
    #     hidden_channels=model_config['hidden_channels'],
    #     num_node_labels=model_config['num_node_labels'],
    #     output_dim=num_classes,
    #     embedding_dim=model_config['embedding_dim']
    # ).to(device)
    
    # 打印模型参数
    print("\n模型参数:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    # 生成模型结构摘要
    model_summary = get_model_summary(model)
    print("\n" + model_summary)
    
    # 创建输出目录
    plots_dir = output_config['plots_dir']
    if plots_dir is None:
        exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'plots_{exp_name}')
    else:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), plots_dir)
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # 提前保存配置文件，包含模型结构
    model_summary_comments = "# " + "\n# ".join(model_summary.split("\n"))
    config_yaml = yaml.dump(config, default_flow_style=False)
    config_with_summary = f"""# GNN Training Configuration and Model Structure
# Generated Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
{model_summary_comments}
# 
# Original Configuration Parameters:
{config_yaml}"""
    
    config_copy_path = os.path.join(plots_dir, 'config.yaml')
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        f.write(config_with_summary)
    
    print(f"配置文件已保存到: {config_copy_path}")
    
    # Learning rate settings from config
    initial_lr = training_config['initial_lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Learning rate decay settings from config
    max_epochs = training_config['max_epochs']
    lr_decay_epochs = training_config['lr_decay_epochs']  
    lr_decay_factor = training_config['lr_decay_factor']
    
    # For tracking metrics
    train_losses = []
    val_metrics = []
    current_lr = initial_lr
    
    # 记录每个epoch的指标，方便后续绘图
    train_metrics_history = []
    val_metrics_history = []
    
    # 梯度统计信息
    gradient_stats = {}
    
    # Training loop
    print("Starting training...")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Learning rate will decrease by factor of {lr_decay_factor} at epochs: {lr_decay_epochs}")
    
    # 早停设置
    early_stopping_config = training_config.get('early_stopping', {'enabled': False})
    early_stop_enabled = early_stopping_config.get('enabled', False)
    early_stop_patience = early_stopping_config.get('patience', 20)
    early_stop_min_delta = early_stopping_config.get('min_delta', 0.001)
    early_stop_monitor = early_stopping_config.get('monitor', 'loss')
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    if early_stop_enabled:
        print(f"Early stopping enabled with patience={early_stop_patience}, min_delta={early_stop_min_delta}, monitoring {early_stop_monitor}")
    
    for epoch in range(max_epochs):
        # Check if we should decrease learning rate
        if epoch in lr_decay_epochs:
            current_lr *= lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"Reducing learning rate to {current_lr}")
        
        # Training phase
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            # loss = model.loss_function(out, data.y)['loss']
            loss.backward()
            
            # 记录梯度
            gradient_stats = log_gradients(model, gradient_stats)
            
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        
        train_loss = total_loss / len(train_subset)
        train_losses.append(train_loss)
        
        # 每10个epoch输出梯度信息
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print_gradient_stats(gradient_stats, epoch)
        
        # Training evaluation
        metrics = evaluate_model(model, train_loader, device, criterion)
        
        # Validation evaluation
        v_metrics = evaluate_model(model, val_loader, device, criterion)
        val_metrics.append(v_metrics)
        
        # 记录本轮的训练和验证指标
        train_metrics_history.append({
            'epoch': epoch,
            'loss': train_loss,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'lr': current_lr
        })
        
        val_metrics_history.append({
            'epoch': epoch,
            'loss': v_metrics['loss'],
            'accuracy': v_metrics['accuracy'],
            'precision': v_metrics['precision'],
            'recall': v_metrics['recall'],
            'f1': v_metrics['f1']
        })
        
        # Print metrics every 10 epochs
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f'Epoch: {epoch:03d}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {v_metrics["loss"]:.4f}, '
                  f'Train Acc: {metrics["accuracy"]:.4f}, Train F1: {metrics["f1"]:.4f}, '
                  f'Val Acc: {v_metrics["accuracy"]:.4f}, Val F1: {v_metrics["f1"]:.4f}')
        
        # 早停检查
        if early_stop_enabled:
            current_val_loss = v_metrics["loss"]
            current_val_acc = v_metrics["accuracy"]
            
            # 根据监控指标决定是否应该更新最佳模型
            should_update = False
            
            if early_stop_monitor == 'loss':
                # 监控损失 - 损失值越小越好
                if current_val_loss < best_val_loss - early_stop_min_delta:
                    should_update = True
                    best_val_loss = current_val_loss
                    improvement = best_val_loss
                    metric_name = "validation loss"
            else:
                # 监控准确率 - 准确率越大越好
                if current_val_acc > best_val_acc + early_stop_min_delta:
                    should_update = True
                    best_val_acc = current_val_acc
                    improvement = best_val_acc
                    metric_name = "validation accuracy"
            
            if should_update:
                patience_counter = 0
                # 保存当前最佳模型状态
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': current_val_loss,
                    'val_accuracy': current_val_acc,
                    'val_f1': v_metrics["f1"]
                }
                print(f"Epoch {epoch}: New best model saved with {metric_name}: {improvement:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    if early_stop_monitor == 'loss':
                        print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                        print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_model_state['epoch']})")
                    else:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                        print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_model_state['epoch']})")
                    break
    
    # 分析梯度变化趋势
    print("\n梯度变化趋势分析:")
    for name, stats in gradient_stats.items():
        # 计算平均梯度
        mean_grads = stats['mean']
        # 计算梯度减小比例
        if len(mean_grads) > 10:  # 确保有足够的数据
            early_grad = np.mean(mean_grads[:10])  # 前10个epoch的平均
            late_grad = np.mean(mean_grads[-10:])  # 后10个epoch的平均
            if early_grad > 0:
                reduction = (early_grad - late_grad) / early_grad
                print(f"{name}: 梯度减小比例 {reduction:.2%}, 初始平均 {early_grad:.8f}, 最终平均 {late_grad:.8f}")
                if late_grad < 1e-4:
                    print(f"  警告: {name} 层的梯度接近于零，可能存在梯度消失问题")
    
    # 如果启用了早停且有最佳模型，则加载最佳模型进行最终评估
    if early_stop_enabled and best_model_state is not None:
        best_epoch = best_model_state['epoch']
        if early_stop_monitor == 'loss':
            print(f"\n加载早停保存的最佳模型（Epoch {best_epoch}，验证损失: {best_model_state['val_loss']:.4f}）")
        else:
            print(f"\n加载早停保存的最佳模型（Epoch {best_epoch}，验证准确率: {best_model_state['val_accuracy']:.4f}）")
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f"最佳模型验证指标 - 准确率: {best_model_state['val_accuracy']:.4f}, F1分数: {best_model_state['val_f1']:.4f}, 损失: {best_model_state['val_loss']:.4f}")
    
    # Final evaluation on validation set
    print("\nFinal Evaluation on Validation Set:")
    final_val_metrics = evaluate_model(model, val_loader, device, criterion)
    print(f"Validation Loss: {final_val_metrics['loss']:.4f}")
    print(f"Validation Accuracy: {final_val_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {final_val_metrics['precision']:.4f}")
    print(f"Validation Recall: {final_val_metrics['recall']:.4f}")
    print(f"Validation F1 Score: {final_val_metrics['f1']:.4f}")
    
    if final_val_metrics['confusion_matrix'] is not None:
        print("\nValidation Confusion Matrix:")
        print(final_val_metrics['confusion_matrix'])
    
    # Make predictions on test set (no evaluation since we don't have labels)
    print("\nGenerating predictions on test set...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            # pred = out['graph_logits'].argmax(dim=1).cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            
            for i in range(len(data.graph_id)):
                predictions.append((int(data.graph_id[i].item()), int(pred[i].item())))
    
    # Save predictions
    result_df = pd.DataFrame(predictions, columns=['graph_id', 'graph_label'])
    result_df.to_csv(output_path, index=False, header=False)
    print(f"Predictions saved to {output_path}")
    
    # 如果配置中指定要保存模型
    if output_config['save_model']:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_config['model_dir'])
        os.makedirs(model_dir, exist_ok=True)
        # 使用相同的时间戳命名模型文件，与结果目录保持一致
        timestamp = plots_dir.split('_')[-1] if '_' in plots_dir else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 如果有最佳模型状态，则保存最佳模型，否则保存最终模型
        if early_stop_enabled and best_model_state is not None:
            best_epoch = best_model_state['epoch']
            model_path = os.path.join(model_dir, f"gnn_model_{timestamp}_best_epoch{best_epoch}.pt")
            torch.save(best_model_state['model_state_dict'], model_path)
            print(f"Best model (epoch {best_epoch}) saved to {model_path}")
            
            # 保存完整检查点，包括优化器状态等
            checkpoint_path = os.path.join(model_dir, f"gnn_checkpoint_{timestamp}_best_epoch{best_epoch}.pt")
            torch.save(best_model_state, checkpoint_path)
            print(f"Complete checkpoint saved to {checkpoint_path}")
        else:
            model_path = os.path.join(model_dir, f"gnn_model_{timestamp}_final.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Final model saved to {model_path}")
    
    
    
    # 生成四图布局
    print("\n生成性能指标图表...")
    
    plt.style.use('ggplot')
    train_df = pd.DataFrame(train_metrics_history)
    val_df = pd.DataFrame(val_metrics_history)
    
    # 将指标数据保存到结果目录
    train_copy_path = os.path.join(plots_dir, 'train_metrics.csv')
    val_copy_path = os.path.join(plots_dir, 'val_metrics.csv')
    train_df.to_csv(train_copy_path, index=False)
    val_df.to_csv(val_copy_path, index=False)
    
    # 创建2x2的图表布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 确定最佳模型所在的epoch
    best_epoch = None
    if early_stop_enabled and best_model_state is not None:
        best_epoch = best_model_state['epoch']
    
    # 1. 左上: 训练和验证损失
    axes[0, 0].plot(train_df['epoch'], train_df['loss'], 'b-', linewidth=1.5, label='Training')
    axes[0, 0].plot(val_df['epoch'], val_df['loss'], 'r-', linewidth=1.5, label='Validation')
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    # 标记最佳模型点
    if best_epoch is not None:
        best_val_loss = val_df[val_df['epoch'] == best_epoch]['loss'].values[0]
        axes[0, 0].plot(best_epoch, best_val_loss, 'ko', markersize=8, label=f'Best Model (Epoch {best_epoch})')
        axes[0, 0].axvline(x=best_epoch, color='k', linestyle='--', alpha=0.3)
        # 添加标注
        axes[0, 0].annotate(f'Best Val Loss: {best_val_loss:.4f}',
                   xy=(best_epoch, best_val_loss),
                   xytext=(best_epoch+5, best_val_loss*1.1),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                   fontsize=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True)
    
    # 2. 右上: 准确率对比
    axes[0, 1].plot(train_df['epoch'], train_df['accuracy'], 'b-', linewidth=1.5, label='Training')
    axes[0, 1].plot(val_df['epoch'], val_df['accuracy'], 'r-', linewidth=1.5, label='Validation')
    axes[0, 1].set_title('Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    # 标记最佳模型点
    if best_epoch is not None:
        best_val_acc = val_df[val_df['epoch'] == best_epoch]['accuracy'].values[0]
        axes[0, 1].plot(best_epoch, best_val_acc, 'ko', markersize=8)
        axes[0, 1].axvline(x=best_epoch, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True)
    
    # 3. 左下: F1分数对比
    axes[1, 0].plot(train_df['epoch'], train_df['f1'], 'b-', linewidth=1.5, label='Training')
    axes[1, 0].plot(val_df['epoch'], val_df['f1'], 'r-', linewidth=1.5, label='Validation')
    axes[1, 0].set_title('F1 Score', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1 Score', fontsize=12)
    # 标记最佳模型点
    if best_epoch is not None:
        best_val_f1 = val_df[val_df['epoch'] == best_epoch]['f1'].values[0]
        axes[1, 0].plot(best_epoch, best_val_f1, 'ko', markersize=8)
        axes[1, 0].axvline(x=best_epoch, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True)
    
    # 4. 右下: 验证精确率和召回率
    axes[1, 1].plot(val_df['epoch'], val_df['precision'], 'g-', linewidth=1.5, label='Precision')
    axes[1, 1].plot(val_df['epoch'], val_df['recall'], 'm-', linewidth=1.5, label='Recall')
    axes[1, 1].set_title('Validation Precision and Recall', fontsize=14)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    # 标记最佳模型点
    if best_epoch is not None:
        best_val_precision = val_df[val_df['epoch'] == best_epoch]['precision'].values[0]
        best_val_recall = val_df[val_df['epoch'] == best_epoch]['recall'].values[0]
        axes[1, 1].plot(best_epoch, best_val_precision, 'ko', markersize=8)
        axes[1, 1].plot(best_epoch, best_val_recall, 'ko', markersize=8)
        axes[1, 1].axvline(x=best_epoch, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(plots_dir, 'metrics_summary.png')
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()
    
    print(f"性能指标图表已保存到: {metrics_plot_path}")
    
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"训练和评估指标数据已保存到: {plots_dir}")
    print(f"性能指标图表已保存到: {metrics_plot_path}")
    print(f"模型总参数量: {total_params:,}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train GNN model on graph classification task')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--plots_dir', type=str, help='Directory name to save plots', default=None)
    args = parser.parse_args()
    
    main(args.config, args.plots_dir)