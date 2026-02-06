# scripts/train_model.py
import torch
import yaml
from ultralytics import YOLO
import argparse
import os

def train_traffic_model(config_path="config/training_config.yaml"):
    """
    Train YOLOv8 model for traffic monitoring
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("TRAINING TRAFFIC MONITORING MODEL")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    if config['model']['pretrained']:
        print("Loading pre-trained YOLOv8n model...")
        model = YOLO('yolov8n.pt')
    else:
        print("Creating new YOLOv8n model...")
        model = YOLO('yolov8n.yaml')
    
    # Training configuration
    train_args = {
        'data': config['data']['dataset_yaml'],
        'epochs': config['training']['epochs'],
        'imgsz': config['model']['input_size'],
        'batch': config['training']['batch_size'],
        'device': device,
        'workers': config['training']['workers'],
        'patience': config['training']['patience'],
        'lr0': config['training']['learning_rate'],
        'lrf': config['training']['final_lr_factor'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'warmup_momentum': config['training']['warmup_momentum'],
        'box': config['loss']['box_loss_weight'],
        'cls': config['loss']['cls_loss_weight'],
        'dfl': config['loss']['dfl_loss_weight'],
        'label_smoothing': config['training']['label_smoothing'],
        'nbs': config['training']['nominal_batch_size'],
        'save': True,
        'save_period': config['training']['save_period'],
        'cache': config['data']['cache'],
        'resume': config['training']['resume'],
        'amp': config['training']['mixed_precision'],
        'fraction': config['data']['fraction'],
        'project': config['training']['project'],
        'name': config['training']['name'],
        'exist_ok': config['training']['exist_ok'],
        'pretrained': config['model']['pretrained'],
        'optimizer': config['training']['optimizer'],
        'seed': config['training']['seed'],
        'deterministic': config['training']['deterministic'],
        'single_cls': config['model']['single_class'],
        'rect': config['data']['rectangular_training'],
        'cos_lr': config['training']['cosine_lr'],
        'close_mosaic': config['augmentation']['close_mosaic'],
        'overlap_mask': config['model']['overlap_mask'],
        'mask_ratio': config['model']['mask_ratio'],
    }
    
    # Add augmentation parameters
    if config['augmentation']['enabled']:
        train_args.update({
            'hsv_h': config['augmentation']['hsv_hue'],
            'hsv_s': config['augmentation']['hsv_saturation'],
            'hsv_v': config['augmentation']['hsv_value'],
            'degrees': config['augmentation']['rotation'],
            'translate': config['augmentation']['translation'],
            'scale': config['augmentation']['scale'],
            'shear': config['augmentation']['shear'],
            'perspective': config['augmentation']['perspective'],
            'flipud': config['augmentation']['flip_vertical'],
            'fliplr': config['augmentation']['flip_horizontal'],
            'mosaic': config['augmentation']['mosaic'],
            'mixup': config['augmentation']['mixup'],
            'copy_paste': config['augmentation']['copy_paste'],
        })
    
    # Start training
    print("\nStarting training...")
    print(f"Dataset: {config['data']['dataset_yaml']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Image size: {config['model']['input_size']}")
    
    results = model.train(**train_args)
    
    # Validate the model
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    
    metrics = model.val(
        data=config['data']['dataset_yaml'],
        imgsz=config['model']['input_size'],
        batch=config['training']['batch_size'],
        device=device,
        split='val'
    )
    
    # Print metrics
    if metrics:
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
    
    return model, results

def export_model(model, config_path="config/training_config.yaml"):
    """
    Export trained model to various formats
    """
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    export_dir = "models"
    os.makedirs(export_dir, exist_ok=True)
    
    export_formats = config['export']['formats']
    
    for fmt in export_formats:
        print(f"\nExporting model to {fmt.upper()} format...")
        
        try:
            if fmt == 'engine':  # TensorRT
                exported_path = model.export(
                    format='engine',
                    imgsz=config['model']['input_size'],
                    device=0 if torch.cuda.is_available() else 'cpu',
                    half=config['export']['half_precision'],
                    workspace=config['export']['workspace_size'],
                    simplify=config['export']['simplify'],
                    opset=config['export']['opset_version'],
                    batch=config['export']['batch_size']
                )
            elif fmt == 'onnx':
                exported_path = model.export(
                    format='onnx',
                    imgsz=config['model']['input_size'],
                    half=config['export']['half_precision'],
                    simplify=config['export']['simplify'],
                    opset=config['export']['opset_version'],
                    dynamic=config['export']['dynamic_axes']
                )
            else:
                exported_path = model.export(format=fmt)
            
            print(f"✓ Model exported to: {exported_path}")
            
        except Exception as e:
            print(f"✗ Failed to export to {fmt}: {str(e)}")
    
    return export_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train traffic monitoring model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    
    args = parser.parse_args()
    
    # Train model
    model, results = train_traffic_model(args.config)
    
    # Export if requested
    if args.export:
        export_model(model, args.config)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)