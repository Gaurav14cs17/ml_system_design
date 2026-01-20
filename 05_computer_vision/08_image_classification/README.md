# 📊 Image Classification

> Transfer learning, fine-tuning strategies, and production deployment

---

## 📑 Table of Contents

1. [Classification Fundamentals](#classification-fundamentals)
2. [Transfer Learning](#transfer-learning)
3. [Fine-Tuning Strategies](#fine-tuning-strategies)
4. [Data Augmentation](#data-augmentation)
5. [Training Best Practices](#training-best-practices)
6. [Production Considerations](#production-considerations)

---

## Classification Fundamentals

![Diagram 1](images/diagram_01.svg)

---

## Transfer Learning

### Why Transfer Learning?

![Diagram 2](images/diagram_02.svg)

### Transfer Learning Strategies

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class TransferLearningClassifier:
    """Transfer learning for image classification."""

    def __init__(self, num_classes, strategy='feature_extract', device='cuda'):
        self.device = device
        self.num_classes = num_classes
        self.strategy = strategy
        self.model = self._build_model()

    def _build_model(self):
        """Build model with transfer learning."""
        # Load pretrained model
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        if self.strategy == 'feature_extract':
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False

        elif self.strategy == 'fine_tune_last':
            # Freeze early layers, train later layers
            for name, param in model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

        elif self.strategy == 'gradual_unfreeze':
            # Start frozen, unfreeze progressively during training
            for param in model.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, self.num_classes)
        )

        return model.to(self.device)

    def unfreeze_layers(self, num_layers):
        """Gradually unfreeze layers for fine-tuning."""
        layers = ['layer4', 'layer3', 'layer2', 'layer1']

        for i, layer_name in enumerate(layers[:num_layers]):
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True

    def get_optimizer_params(self, lr_backbone=1e-5, lr_head=1e-3):
        """Get parameter groups with different learning rates."""
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        return [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_head}
        ]

```

---

## Fine-Tuning Strategies

![Diagram 3](images/diagram_03.svg)

---

## Data Augmentation

### Augmentation Strategies

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=224):
    """Strong augmentation for training."""
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.5),
            A.GridDistortion(num_steps=5),
            A.ElasticTransform(alpha=1, sigma=50),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=4.0),
            A.Equalize(),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    """Minimal transforms for validation."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

```

### Advanced Augmentation

![Diagram 4](images/diagram_04.svg)

```python
import numpy as np
import torch

def mixup(images, labels, alpha=0.2):
    """Mixup augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]

    return mixed_images, labels_a, labels_b, lam

def cutmix(images, labels, alpha=1.0):
    """CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    H, W = images.size(2), images.size(3)

    # Get random box
    cut_ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cutmix
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda for actual area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))

    return mixed_images, labels, labels[index], lam

```

---

## Training Best Practices

### Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm

class ImageClassificationTrainer:
    """Production-ready image classification trainer."""

    def __init__(self, model_name='resnet50', num_classes=10, device='cuda'):
        self.device = device
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        ).to(device)

        self.scaler = GradScaler()  # For mixed precision

    def train_epoch(self, dataloader, optimizer, criterion, use_mixup=True):
        """Train for one epoch with mixed precision and augmentation."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Optional mixup
            if use_mixup and np.random.random() > 0.5:
                images, labels_a, labels_b, lam = mixup(images, labels)

                with autocast():
                    outputs = self.model(images)
                    loss = lam * criterion(outputs, labels_a) + \
                           (1 - lam) * criterion(outputs, labels_b)
            else:
                with autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

            # Backward with mixed precision
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            'accuracy': 100. * correct / total,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, train_loader, val_loader, epochs=100, lr=1e-4):
        """Full training loop with learning rate scheduling."""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_acc = 0

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            val_metrics = self.evaluate(val_loader)
            scheduler.step()

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
            print(f"  Val: Acc={val_metrics['accuracy']:.2f}%")

            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"  New best model saved!")

        return best_acc

```

---

## Production Considerations

### Model Selection for Production

| Model | Accuracy | Latency | Memory | Best For |
|-------|----------|---------|--------|----------|
| MobileNetV3 | Good | Very Fast | Low | Mobile/Edge |
| EfficientNet-B0 | Good | Fast | Low | Balanced |
| ResNet-50 | Very Good | Medium | Medium | General |
| ViT-B/16 | Excellent | Slow | High | Cloud/Accuracy |

### Inference Optimization

```python
import torch
import onnxruntime as ort

class ProductionClassifier:
    """Optimized classifier for production."""

    def __init__(self, model_path, device='cuda'):
        self.device = device

        # Try ONNX Runtime first (faster)
        if model_path.endswith('.onnx'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.use_onnx = True
        else:
            self.model = torch.jit.load(model_path)
            self.model.to(device).eval()
            self.use_onnx = False

    def preprocess(self, image):
        """Preprocess image for inference."""
        import cv2
        import numpy as np

        # Resize
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Transpose to CHW and add batch dimension
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, 0)

        return image.astype(np.float32)

    @torch.no_grad()
    def predict(self, image, top_k=5):
        """Run inference and return top-k predictions."""
        preprocessed = self.preprocess(image)

        if self.use_onnx:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: preprocessed})
            logits = outputs[0][0]
        else:
            tensor = torch.from_numpy(preprocessed).to(self.device)
            logits = self.model(tensor)[0].cpu().numpy()

        # Softmax
        probs = np.exp(logits) / np.exp(logits).sum()

        # Top-k
        top_indices = probs.argsort()[-top_k:][::-1]

        return [
            {'class_id': int(idx), 'confidence': float(probs[idx])}
            for idx in top_indices
        ]

```

---

## 📚 Key Takeaways

1. **Transfer learning** almost always beats training from scratch
2. **Gradual unfreezing** and **discriminative LR** improve fine-tuning
3. **Strong augmentation** (mixup, cutmix) helps generalization
4. **Label smoothing** prevents overconfidence
5. **Mixed precision** speeds up training with no accuracy loss
6. **ONNX/TorchScript** for production inference

---

## 🔗 Next Steps

- [Face Recognition →](../09_face_recognition/) - Specialized classification

- [Deployment →](../15_deployment/) - Production optimization

---

*Classification is the foundation - master it before tackling complex tasks.* 🎯

---

<div align="center">

**[⬆ Back to Top](#)** | **[📚 Main Repository](https://github.com/Gaurav14cs17/ml_system_design)**

Made with 💜 by [Gaurav14cs17](https://github.com/Gaurav14cs17)

</div>
