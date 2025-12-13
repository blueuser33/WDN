import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class ModelTrainer:
    """通用模型训练器"""

    def __init__(self, model, device, optimizer=None, criterion=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)
        self.criterion = criterion if criterion else nn.MSELoss()
        self.best_val_loss = float('inf')
        self.train_losses = []  # ✅ 新增
        self.val_losses = []  # ✅ 新增

    def train_one_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc="Training", leave=False):
            x, y = x.to(self.device).float(), y.to(self.device).float()
            self.optimizer.zero_grad()

            output = self.model(x)
            loss = self.criterion(output, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device).float(), y.to(self.device).float()
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs, model_save_path):
        """完整训练流程"""
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_save_path)
                print(f"✅ 保存最佳模型至 {model_save_path}")

        # ✅ 保存 loss 曲线数据
        np.save(model_save_path.replace(".pth", "_train_loss.npy"), np.array(self.train_losses))
        np.save(model_save_path.replace(".pth", "_val_loss.npy"), np.array(self.val_losses))
        print("📊 已保存训练与验证loss曲线数据")

    def test(self, test_loader, model_path):
        """测试模型性能"""
        self.model.load_state_dict(torch.load(model_path))
        test_loss = self.evaluate(test_loader)
        print(f"Final Test Loss: {test_loss:.6f}")
        return test_loss

    def predict(self, data_loader):
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)  # [B, n_pred, N]
                preds.append(out.cpu())
                trues.append(y.cpu())
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        return preds, trues
