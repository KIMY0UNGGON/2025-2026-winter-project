import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장용
import matplotlib.pyplot as plt
from datetime import datetime

class TokenTrainer:
    def __init__(self, model, optimizer, device, output_dir, evaluator=None, val_data_path=None, scheduler=None, hyperparams=None, accelerator=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.val_data_path = val_data_path
        self.scheduler = scheduler
        self.hyperparams = hyperparams or {}
        self.best_val_f1 = 0.0
        self.patience = 4
        self.accelerator = accelerator

        ## =============================================
        ## [가속화] CUDA 백엔드 최적화 설정
        ## =============================================
        if torch.cuda.is_available():
            ## [가속화] cuDNN Benchmark — 입력 크기가 고정일 때 최적 알고리즘 자동 선택
            torch.backends.cudnn.benchmark = True
            ## [가속화] TF32 행렬곱 — Ampere+ GPU에서 FP32 연산을 TF32로 가속 (정밀도 미세 손실)
            torch.backends.cuda.matmul.allow_tf32 = True
            ## [가속화] TF32 cuDNN — cuDNN 연산도 TF32 허용
            torch.backends.cudnn.allow_tf32 = True
            print("  [가속화] cuDNN Benchmark: ON | TF32 MatMul: ON | TF32 cuDNN: ON")
        
        # 클래스 가중치 설정: O=1.0, B-HAL=5.0, I-HAL=1.5
        class_weights = torch.tensor([1.0, 5.0, 1.5]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, 
            ignore_index=-100,
            label_smoothing=0.1
        )
        print(f"   Class weights applied: O=1.0, B-HAL=5.0, I-HAL=1.5 | Label Smoothing: 0.1")
        
        # 학습 히스토리 기록용
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "train_precision": [],
            "train_recall": [],
            "val_loss": [],
            "val_f1": [],
            "lr": []
        }

    def train(self, train_loader, epochs, val_loader=None):
        # Accelerate가 모델을 이미 올바른 디바이스에 배치함
        print("Training Started...")
        if self.accelerator:
            print(f"  Using Accelerate | Mixed Precision: {self.accelerator.mixed_precision} | Grad Accum Steps: {self.accelerator.gradient_accumulation_steps}")
        early_stop_counter = 0
        
        for ep in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # 평가지표용 리스트
            all_preds = []
            all_labels = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {ep+1} Training")
            for batch in pbar:
                # accelerator.accumulate()가 gradient accumulation을 자동 관리
                with self.accelerator.accumulate(self.model):
                    # Accelerate가 이미 배치를 올바른 디바이스로 이동시킴
                    with self.accelerator.autocast():
                        outputs = self.model(**batch)
                        # 커스텀 가중치 loss 계산 (모델 기본 loss 대신)
                        logits = outputs.logits  # (batch_size, seq_len, num_labels)
                        labels = batch["labels"]  # (batch_size, seq_len)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    # Accelerate가 mixed precision backward + gradient accumulation 자동 처리
                    self.accelerator.backward(loss)
                    
                    # Gradient Clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                
                # TQDM 실시간 정보 업데이트 (현재 LR 표시)
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})
                
                # 예측값 및 레이블 추출 (성능 평가용)
                preds = torch.argmax(logits, dim=-1)
                labels = batch["labels"]
                
                # 배치 데이터를 numpy로 변환 및 평탄화 (GPU -> CPU)
                preds = preds.view(-1).cpu().numpy()
                labels = labels.view(-1).cpu().numpy()
                
                # 패딩 토큰(-100)을 제외하고 유효한 라벨만 필터링
                valid_mask = labels != -100
                all_preds.extend(preds[valid_mask])
                all_labels.extend(labels[valid_mask])
            
            avg_loss = total_loss / num_batches
            
            # Epoch 단위의 정확도(Acc) 및 F1 스코어 계산 (Hallucination 탐지이므로 macro f1 권장)
            acc = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[1, 2], average='micro', zero_division=0)
            
            # Validation 평가 (Loss + F1)
            val_loss = None
            val_f1 = None
            if val_loader:
                self.model.eval()
                val_total_loss = 0
                val_preds = []
                val_labels = []
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {ep+1} Validation"):
                        with self.accelerator.autocast():
                            outputs = self.model(**batch)
                            # 커스텀 가중치 loss 계산
                            v_logits = outputs.logits
                            v_loss = self.criterion(v_logits.view(-1, v_logits.size(-1)), batch["labels"].view(-1))
                        val_total_loss += v_loss.item()
                        
                        # Val F1 계산용 예측값 수집
                        v_preds = torch.argmax(outputs.logits, dim=-1).view(-1).cpu().numpy()
                        v_labels = batch["labels"].view(-1).cpu().numpy()
                        v_valid = v_labels != -100
                        val_preds.extend(v_preds[v_valid])
                        val_labels.extend(v_labels[v_valid])
                
                val_loss = val_total_loss / len(val_loader)
                _, _, val_f1, _ = precision_recall_fscore_support(
                    val_labels, val_preds, labels=[1, 2], average='micro', zero_division=0
                )
            
            # 히스토리 기록
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(acc)
            self.history["train_f1"].append(f1)
            self.history["train_precision"].append(precision)
            self.history["train_recall"].append(recall)
            self.history["val_loss"].append(val_loss if val_loss is not None else None)
            self.history["val_f1"].append(val_f1 if val_f1 is not None else None)
            self.history["lr"].append(current_lr)
            
            # 결과 출력
            val_str = ""
            if val_loss is not None:
                val_str = f" | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
            print(f"Epoch {ep+1} | LR: {current_lr:.2e} | Loss: {avg_loss:.4f}{val_str} | Train Acc: {acc:.4f} | Train F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
            
            # 모델 저장 및 Early Stopping (Val F1 기준)
            if val_f1 is not None:
                if val_f1 > self.best_val_f1:
                    self._save_checkpoint(self.output_dir, ep + 1, val_f1, val_loss, avg_loss, f1)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f" >>> Val F1 did not improve. Early stopping counter: {early_stop_counter}/{self.patience}")
                    if early_stop_counter >= self.patience:
                        print(f"Early Stopping triggered at Epoch {ep+1}!")
                        break
            else:
                # Validation이 없는 경우 매 에폭 저장
                self._save_checkpoint(self.output_dir, ep + 1, 0, 0, avg_loss, f1)
            
        print("Training complete.")
        
        # 마지막 모델 저장 (Final 모델)
        final_dir = os.path.join(self.output_dir, "final")
        print(f" >>> Saving final model to {final_dir}...")
        self._save_checkpoint(final_dir, epochs, self.best_val_f1, 0, avg_loss, f1)
        
        # 학습 히스토리 저장 및 시각화
        self._save_and_plot_history()

    def _save_checkpoint(self, save_dir, epoch, val_f1, val_loss, train_loss, train_f1):
        """모델, 옵티마이저, 메타데이터 등을 지정된 디렉토리에 저장합니다."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 모델 저장 (Accelerate unwrap)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_dir)
        
        # 2. 학습 상태 저장
        training_state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler:
            training_state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        
        # 3. 메타 데이터 저장
        meta = {
            "epoch": epoch,
            "best_f1": self.best_val_f1,
            "curr_val_f1": val_f1,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "hyperparameters": self.hyperparams,
            "config": {
                "best_threshold": 0.5
            },
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(save_dir, "training_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=3, ensure_ascii=False)

    def _save_and_plot_history(self):
        """학습 히스토리를 JSON으로 저장하고, 성능 변화 그래프를 생성합니다."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # JSON 히스토리 저장
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
        print(f"  학습 히스토리 저장 완료: {history_path}")
        
        # 그래프 생성
        epochs_range = list(range(1, len(self.history["train_loss"]) + 1))
        if not epochs_range:
            print("   기록된 에폭이 없어 그래프를 생성할 수 없습니다.")
            return
        
        has_val = any(v is not None for v in self.history["val_loss"])
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Token Hallucination Detection — Training Report', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        colors = {
            'train': '#2196F3',       # Blue
            'val': '#FF5722',         # Deep Orange
            'acc': '#4CAF50',         # Green
            'precision': '#9C27B0',   # Purple
            'recall': '#FF9800',      # Orange
            'lr': '#607D8B',          # Blue Grey
        }
        
        # ── 1. Loss 곡선 ──
        ax1 = axes[0, 0]
        ax1.plot(epochs_range, self.history["train_loss"], 
                 marker='o', color=colors['train'], linewidth=2, markersize=6, label='Train Loss')
        if has_val:
            val_losses = [v for v in self.history["val_loss"] if v is not None]
            val_epochs = [e for e, v in zip(epochs_range, self.history["val_loss"]) if v is not None]
            ax1.plot(val_epochs, val_losses, 
                     marker='s', color=colors['val'], linewidth=2, markersize=6, label='Val Loss')
        ax1.set_title('Loss', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(fontsize=10)
        ax1.set_xticks(epochs_range)
        
        # ── 2. F1 Score 곡선 ──
        ax2 = axes[0, 1]
        ax2.plot(epochs_range, self.history["train_f1"], 
                 marker='o', color=colors['train'], linewidth=2, markersize=6, label='Train F1')
        if has_val:
            val_f1s = [v for v in self.history["val_f1"] if v is not None]
            val_f1_epochs = [e for e, v in zip(epochs_range, self.history["val_f1"]) if v is not None]
            ax2.plot(val_f1_epochs, val_f1s, 
                     marker='s', color=colors['val'], linewidth=2, markersize=6, label='Val F1')
            # Best F1 표시
            if val_f1s:
                best_idx = np.argmax(val_f1s)
                ax2.annotate(f'Best: {val_f1s[best_idx]:.4f}', 
                            xy=(val_f1_epochs[best_idx], val_f1s[best_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, fontweight='bold', color=colors['val'],
                            arrowprops=dict(arrowstyle='->', color=colors['val']))
        ax2.set_title('F1 Score (Micro, Labels=[1,2])', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend(fontsize=10)
        ax2.set_xticks(epochs_range)
        ax2.set_ylim(0, 1.05)
        
        # ── 3. Precision / Recall / Accuracy 곡선 ──
        ax3 = axes[1, 0]
        ax3.plot(epochs_range, self.history["train_acc"], 
                 marker='o', color=colors['acc'], linewidth=2, markersize=6, label='Accuracy')
        ax3.plot(epochs_range, self.history["train_precision"], 
                 marker='^', color=colors['precision'], linewidth=2, markersize=6, label='Precision')
        ax3.plot(epochs_range, self.history["train_recall"], 
                 marker='v', color=colors['recall'], linewidth=2, markersize=6, label='Recall')
        ax3.set_title('Train Metrics', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend(fontsize=10)
        ax3.set_xticks(epochs_range)
        ax3.set_ylim(0, 1.05)
        
        # ── 4. Learning Rate 곡선 ──
        ax4 = axes[1, 1]
        ax4.plot(epochs_range, self.history["lr"], 
                 marker='D', color=colors['lr'], linewidth=2, markersize=6, label='Learning Rate')
        ax4.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend(fontsize=10)
        ax4.set_xticks(epochs_range)
        ax4.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        # 레이아웃 및 저장
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_dir, f"training_plot_{timestamp}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"   학습 성능 그래프 저장 완료: {plot_path}")
