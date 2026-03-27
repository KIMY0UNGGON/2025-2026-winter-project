## =============================================
## 1. Imports
## =============================================
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_cosine_schedule_with_warmup, DataCollatorForTokenClassification
from accelerate import Accelerator

from token_data import load_jsonl_lazy, TokenHalDataset
from train_class_learning import TokenTrainer
from evaluator_module import TokenEvaluator

LABEL_LIST = ["O", "B-HAL", "I-HAL"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}https://github.com/LimKH03/2025-2026-winter-project/blob/main/train_token_hallucination.py

## =============================================
## 2. Hyperparameters
## =============================================
MODEL_NAME = "answerdotai/ModernBERT-base"
EPOCHS = 10
LR = 2e-5
BATCH_SIZE = 32
MAX_LENGTH = 8192

ACC_STEPS = 2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, "psiloqa_data", "train.jsonl")
VAL_PATH = os.path.join(SCRIPT_DIR, "psiloqa_data", "validation.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "token_hal_model_v1_restored")

## =============================================
## 3. Model & Data Setup
## =============================================
def build_model_and_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_LIST))

    ## [가속화] Gradient Checkpointing — VRAM 절감 (속도 약간 감소 대신 메모리 대폭 절약)
    model.gradient_checkpointing_enable()
    print("  [가속화] Gradient Checkpointing: ON")

    train_samples = load_jsonl_lazy(TRAIN_PATH)
    val_samples = load_jsonl_lazy(VAL_PATH)
    
    if not train_samples:
        print("Train data not found. Exiting.")
        return None, None, None, None

    train_cache_path = os.path.join(SCRIPT_DIR, "psiloqa_data", f"train_cache_dp_{MAX_LENGTH}.pt")
    val_cache_path = os.path.join(SCRIPT_DIR, "psiloqa_data", f"val_cache_dp_{MAX_LENGTH}.pt")

    train_dataset = TokenHalDataset(train_samples, tokenizer, max_length=MAX_LENGTH, cache_path=train_cache_path)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    
    ## [가속화] DataLoader 최적화
    ##   - pin_memory=True : CPU→GPU 전송 속도 향상 (CUDA 전용)
    ##   - num_workers=4   : 멀티프로세스 데이터 로딩 (I/O 병목 해소)
    ##   - prefetch_factor=2: 각 워커가 미리 2배치씩 준비
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=data_collator,
        pin_memory=True, num_workers=4, prefetch_factor=2,
        persistent_workers=True,  ## [가속화] 워커 프로세스 재활용 (에폭 간 재시작 방지)
    )

    val_loader = None
    if val_samples:
        val_dataset = TokenHalDataset(val_samples, tokenizer, max_length=MAX_LENGTH, cache_path=val_cache_path)
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=data_collator,
            pin_memory=True, num_workers=4, prefetch_factor=2,
            persistent_workers=True,
        )

    return model, tokenizer, train_loader, val_loader

## =============================================
## 4. Training
## =============================================
def train():
    print("Running modularized token hallucination training script")
    print(f"  Mixed Precision: bf16 | Gradient Accumulation Steps: {ACC_STEPS}")

    # Accelerator 초기화 (bf16 mixed precision + gradient accumulation)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=ACC_STEPS,
    )

    model, tokenizer, train_loader, val_loader = build_model_and_data()
    if model is None:
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Scheduler 설정 (Warmup 15%)
    # effective steps = ceil(len(train_loader) / ACC_STEPS) per epoch
    num_update_steps_per_epoch = -(-len(train_loader) // ACC_STEPS)  # ceiling division
    num_training_steps = num_update_steps_per_epoch * EPOCHS
    num_warmup_steps = int(num_training_steps * 0.15)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    # Accelerate로 모델, 옵티마이저, 데이터로더, 스케줄러 래핑
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    device = accelerator.device
    evaluator = TokenEvaluator(model=model, tokenizer=tokenizer, device=device)

    # 하이퍼파라미터 기록용 딕셔너리
    hyperparams = {
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * ACC_STEPS,
        "gradient_accumulation_steps": ACC_STEPS,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "max_length": MAX_LENGTH,

        "optimizer": type(optimizer).__name__,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.15,
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
    }

    trainer = TokenTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        output_dir=OUTPUT_DIR,
        evaluator=evaluator,
        val_data_path=VAL_PATH,
        scheduler=scheduler,
        hyperparams=hyperparams,
        accelerator=accelerator,
    )

    trainer.train(train_loader=train_loader, epochs=EPOCHS, val_loader=val_loader)


if __name__ == "__main__":
    train()
