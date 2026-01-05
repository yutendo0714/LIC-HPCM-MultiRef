"""
Phase 1訓練スクリプトの例

使用方法:
    python examples/train_phase1_example.py --config config/phase1.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
sys.path.insert(0, '/workspace/LIC-HPCM-MultiRef')

from src.models.multiref.phase1 import HPCM_MultiRef_Phase1


class RDLoss(nn.Module):
    """Rate-Distortion Loss"""
    def __init__(self, lmbda=0.01):
        super().__init__()
        self.lmbda = lmbda
    
    def forward(self, output, target):
        # Distortion (MSE)
        mse_loss = nn.functional.mse_loss(output['x_hat'], target)
        
        # Rate (negative log-likelihood)
        y_likelihoods = output['likelihoods']['y']
        z_likelihoods = output['likelihoods']['z']
        
        # Bits per pixel
        num_pixels = target.size(0) * target.size(2) * target.size(3)
        y_bpp = torch.log(y_likelihoods).sum() / (-num_pixels)
        z_bpp = torch.log(z_likelihoods).sum() / (-num_pixels)
        bpp = y_bpp + z_bpp
        
        # Total loss
        loss = self.lmbda * 255**2 * mse_loss + bpp
        
        return {
            'loss': loss,
            'mse': mse_loss,
            'bpp': bpp,
            'y_bpp': y_bpp,
            'z_bpp': z_bpp
        }


def train_epoch(model, dataloader, optimizer, criterion, epoch, device):
    """1エポックの訓練"""
    model.train()
    
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        output = model(images, training=True)
        
        # Loss計算
        loss_dict = criterion(output, images)
        loss = loss_dict['loss']
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # 統計
        total_loss += loss.item()
        total_mse += loss_dict['mse'].item()
        total_bpp += loss_dict['bpp'].item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} MSE: {loss_dict['mse'].item():.4f} "
                  f"BPP: {loss_dict['bpp'].item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_bpp = total_bpp / len(dataloader)
    
    return {
        'loss': avg_loss,
        'mse': avg_mse,
        'bpp': avg_bpp
    }


def validate(model, dataloader, criterion, device):
    """検証"""
    model.eval()
    
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            
            output = model(images, training=False)
            loss_dict = criterion(output, images)
            
            total_loss += loss_dict['loss'].item()
            total_mse += loss_dict['mse'].item()
            total_bpp += loss_dict['bpp'].item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_bpp = total_bpp / len(dataloader)
    
    return {
        'loss': avg_loss,
        'mse': avg_mse,
        'bpp': avg_bpp
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_multiref', type=bool, default=True)
    parser.add_argument('--max_refs', type=int, default=4)
    parser.add_argument('--topk_refs', type=int, default=2)
    parser.add_argument('--compress_ratio', type=int, default=4)
    parser.add_argument('--lmbda', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデル構築
    print("\nBuilding model...")
    model = HPCM_MultiRef_Phase1(
        M=320,
        N=256,
        enable_multiref=args.enable_multiref,
        max_refs=args.max_refs,
        topk_refs=args.topk_refs,
        compress_ratio=args.compress_ratio
    ).to(device)
    
    # 損失関数とオプティマイザ
    criterion = RDLoss(lmbda=args.lmbda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # データローダー（ダミー実装 - 実際にはデータセットを用意）
    # TODO: 実際のデータセットに置き換え
    print("\nNote: This is a template. Replace with actual dataset.")
    print("Example dataset setup:")
    print("  from torchvision.datasets import ImageFolder")
    print("  from torchvision import transforms")
    print("  dataset = ImageFolder(root='path/to/dataset', transform=transforms.ToTensor())")
    print("  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)")
    
    # 訓練ループ（テンプレート）
    print("\nTraining template ready. Add your dataset to start training.")
    print(f"Configuration:")
    print(f"  - Multi-Reference: {args.enable_multiref}")
    print(f"  - max_refs: {args.max_refs}")
    print(f"  - topk_refs: {args.topk_refs}")
    print(f"  - lambda: {args.lmbda}")
    print(f"  - learning rate: {args.lr}")
    
    """
    # 実際の訓練ループ（データセット準備後に有効化）
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # 訓練
        train_stats = train_epoch(model, train_loader, optimizer, criterion, epoch, device)
        print(f"Train - Loss: {train_stats['loss']:.4f}, MSE: {train_stats['mse']:.4f}, BPP: {train_stats['bpp']:.4f}")
        
        # 検証
        val_stats = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_stats['loss']:.4f}, MSE: {val_stats['mse']:.4f}, BPP: {val_stats['bpp']:.4f}")
        
        # Learning rate調整
        scheduler.step(val_stats['loss'])
        
        # チェックポイント保存
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_stats': train_stats,
                'val_stats': val_stats,
            }, f'checkpoints/phase1_epoch{epoch}.pth')
    """


if __name__ == '__main__':
    main()
