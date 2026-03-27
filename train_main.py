## =============================================
## 1. Imports
## =============================================

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_cosine_schedule_with_warmup, DataCollatorForTokenClassification
from accelerate import Accelerator

from PsiloQA_Dataset import TokenHalDataset
from train_class_learning import TokenTrainer
from evaluator_module import TokenEvaluator
from InteractionLayer import ModernBertWithTokenMatch


## =============================================
## 2. Hyperparameters
## =============================================
MODEL_NAME = "answerdotai/ModernBERT-base"
EPOCHS = 10
LR = 2e-5
BATCH_SIZE = 32
MAX_LENGTH = 8192

ACC_STEPS = 2
SCRIPT_DIR ="script"
OUTPUT_DIR = "model_save"

## =============================================
## 3. Model & Data Setup
## =============================================
def build_model_and_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ModernBertWithTokenMatch.from_pretrained(MODEL_NAME, num_labels=2)
    model.set_token_match(sep_token_id=tokenizer.eos_token_id)

    ## [가속화] Gradient Checkpointing — VRAM 절감 (속도 약간 감소 대신 메모리 대폭 절약)
    model.gradient_checkpointing_enable()
    print("  [가속화] Gradient Checkpointing: ON")
   


    train_dataset = TokenHalDataset(tokenizer, max_length=MAX_LENGTH, split="train")
    val_dataset = TokenHalDataset(tokenizer, max_length=MAX_LENGTH, split="validation")
    test_dataset = TokenHalDataset(tokenizer, max_length=MAX_LENGTH, split="test")
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    
    ## [가속화] DataLoader 최적화
    ##   - pin_memory=True : CPU→GPU 전송 속도 향상 (CUDA 전용)
    ##   - num_workers=4   : 멀티프로세스 데이터 로딩 (I/O 병목 해소)
    ##   - prefetch_factor=2: 각 워커가 미리 2배치씩 준비
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=data_collator,
        pin_memory=True, num_workers=8, prefetch_factor=2,
        persistent_workers=True,  ## [가속화] 워커 프로세스 재활용 (에폭 간 재시작 방지)
    )


    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=data_collator,
        pin_memory=True, num_workers=8, prefetch_factor=2,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=data_collator,
        pin_memory=True, num_workers=4, prefetch_factor=2,
        persistent_workers=True,
    )


    return model, tokenizer, train_loader, val_loader, test_loader

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

    model, tokenizer, train_loader, val_loader,test_loader = build_model_and_data()
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
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )

    device = accelerator.device

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
        scheduler=scheduler,
        hyperparams=hyperparams,
        accelerator=accelerator,
    )

    trainer.train(train_loader=train_loader, epochs=EPOCHS, val_loader=val_loader)


if __name__ == "__main__":
    train()
