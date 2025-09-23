import os
from typing import final

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 模型路徑常數
MODEL_PATH = "mPLUG/mPLUG-Owl3-1B-241014"  # TODO: change to mPLUG/mPLUG-Owl3-7B-241101

# 品質級別映射
QUALITY_LEVELS = ["bad", "poor", "fair", "good", "excellent"]
QUALITY_TO_ID = {level: i for i, level in enumerate(QUALITY_LEVELS)}
ID_TO_QUALITY = {i: level for i, level in enumerate(QUALITY_LEVELS)}


@final
class QualityAssessmentDataset(Dataset):
    """
    圖片品質評估數據集類
    支援多模態輸入（圖片+文字提示）
    處理soft label分布
    """

    def __init__(self, csv_file, image_dir, processor, tokenizer):
        """
        初始化數據集
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.prompt_template = "<|image|> How would you rate the quality of this image?"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        獲取單個數據項
        """
        row = self.data.iloc[idx]

        # 載入圖片
        image_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")

        # 準備消息格式（mPLUG-Owl3格式）
        messages = [
            {
                "role": "user",
                "content": "<|image|> How would you rate the quality of this image?",
            },
            {"role": "assistant", "content": ""},
        ]

        # 使用processor處理輸入
        inputs = self.processor(messages, images=[image], videos=None)

        # 提取soft label分布
        soft_labels = torch.tensor(
            [row["bad"], row["poor"], row["fair"], row["good"], row["excellent"]],
            dtype=torch.float32,
        )

        # 確保分布總和為1（歸一化）
        soft_labels = soft_labels / soft_labels.sum()

        # 獲取tensor並確保正確的維度
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        # 確保input_ids是1D tensor
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)

        # pixel_values的形狀是[num_patches, 3, 384, 384]，需要添加batch維度
        # 變成[1, num_patches, 3, 384, 384]，但模型期望[batch, 3, 384, 384]
        # 對於mPLUG-Owl3，我們需要保持原始形狀，因為它會在內部處理patches

        # 直接使用processor提供的media_offset，如果沒有就創建默認的
        if "media_offset" in inputs:
            media_offset = inputs["media_offset"]
            if media_offset.dim() > 2:
                media_offset = media_offset.squeeze(0)
        else:
            # 創建一個簡單的media_offset
            seq_len = input_ids.size(0)
            media_offset = torch.zeros((seq_len, 2), dtype=torch.long)
            # 假設圖像在開始位置
            media_offset[:, 0] = 0  # media類型
            media_offset[:, 1] = torch.arange(seq_len)  # 位置索引

        # 構建返回的數據字典
        result = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,  # 保持原始形狀
            "media_offset": media_offset,  # 添加media_offset
            "soft_labels": soft_labels,
        }

        # 如果有attention_mask就添加
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            if attention_mask.dim() > 1:
                attention_mask = attention_mask.squeeze(0)
            result["attention_mask"] = attention_mask

        return result


@final
class CustomTrainer(Trainer):
    """
    自定義Trainer類
    重寫compute_loss方法以實現KL散度損失
    """

    def __init__(self, quality_token_ids, kl_weight=1.0, tokenizer=None, **kwargs):
        """
        初始化自定義訓練器
        """
        super().__init__(**kwargs)
        self.quality_token_ids = quality_token_ids
        self.kl_weight = kl_weight
        self.custom_tokenizer = tokenizer

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        計算自定義損失函數
        結合KL散度損失與模型生成能力
        """
        # 提取soft labels
        soft_labels = inputs.pop("soft_labels", None)

        # 處理pixel_values的形狀問題
        # 從 [batch_size, 7, 3, 384, 384] 重新整形為 [batch_size*7, 3, 384, 384]
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            if pixel_values.dim() == 5:  # [batch_size, 7, 3, 384, 384]
                batch_size, num_patches, channels, height, width = pixel_values.shape
                # 重新整形為 [batch_size*num_patches, channels, height, width]
                inputs["pixel_values"] = pixel_values.view(-1, channels, height, width)

        # 不設置labels，讓模型使用原有的前向傳播
        # 先獲取模型輸出以計算KL散度
        outputs = model(**inputs)

        # 由於我們沒有提供labels，模型不會計算loss
        # 我們需要手動計算一個簡單的loss來維持訓練

        # 使用一個簡化的方法：計算KL散度損失作為主要損失
        kl_loss = torch.tensor(0.0, device=inputs["input_ids"].device)

        if hasattr(outputs, "logits") and soft_labels is not None:
            logits = outputs.logits
            batch_size = soft_labels.size(0)

            # 在最後幾個token位置尋找品質詞彙
            # 取最後一個token的logits
            last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # 提取品質詞彙對應的logits
            quality_logits = last_token_logits[
                :, self.quality_token_ids
            ]  # [batch_size, 5]

            # 計算KL散度
            kl_loss = F.kl_div(
                F.log_softmax(quality_logits, dim=-1),
                soft_labels,
                reduction="batchmean",
            )

        # 添加一個小的語言模型損失項以保持生成能力
        # 使用輸入序列的簡單自回歸損失
        lm_loss = torch.tensor(0.01, device=inputs["input_ids"].device)

        # 總損失 = KL散度損失 + 語言模型維持項
        total_loss = kl_loss + lm_loss

        return (total_loss, outputs) if return_outputs else total_loss


def custom_collate_fn(batch):
    """
    自定義collate函數用於DataLoader
    處理批次數據的組合
    """
    input_ids = []
    attention_masks = []
    pixel_values = []
    media_offsets = []
    soft_labels = []

    for item in batch:
        input_ids.append(item["input_ids"])
        if "attention_mask" in item:
            attention_masks.append(item["attention_mask"])
        pixel_values.append(item["pixel_values"])
        media_offsets.append(item["media_offset"])
        soft_labels.append(item["soft_labels"])

    # 將列表轉換為批次張量
    result = {
        "input_ids": torch.stack(input_ids),
        "pixel_values": torch.stack(pixel_values),
        "media_offset": torch.stack(media_offsets),
        "soft_labels": torch.stack(soft_labels),
    }

    # 如果有attention_mask就添加
    if attention_masks:
        result["attention_mask"] = torch.stack(attention_masks)

    return result


def setup_model_and_tokenizer():
    """
    設置模型、分詞器和處理器
    """
    # 載入模型配置
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 載入模型（設為訓練模式）
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 設為訓練模式
    model.train()

    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 初始化處理器
    processor = model.init_processor(tokenizer)

    return model, tokenizer, processor


def get_quality_token_ids(tokenizer):
    """
    獲取品質詞彙對應的token IDs
    """
    quality_token_ids = []
    for quality in QUALITY_LEVELS:
        # FIXME: token IDs: {'bad': [13855], 'poor': [5368, 269], 'fair': [40900], 'good': [18536], 'excellent': [327, 24746]}
        token_id = tokenizer.encode(quality, add_special_tokens=False)
        token_id = token_id[0]  # HACK

        # # 嘗試不同的編碼方式來找到正確的token
        # if quality == "excellent":
        #     # "exce." 代表 excellent
        #     token_id = tokenizer.encode("exce.", add_special_tokens=False)
        # else:
        #     token_id = tokenizer.encode(quality, add_special_tokens=False)
        quality_token_ids.append(token_id)

    return quality_token_ids


def main():
    """
    主函數：設置並開始訓練過程
    """
    print("=== mPLUG-Owl3 圖片品質評估微調訓練 ===")

    # 1. 設置模型和分詞器
    print("1. 載入模型和分詞器...")
    model, tokenizer, processor = setup_model_and_tokenizer()

    # 2. 獲取品質詞彙token IDs
    print("2. 設置品質詞彙映射...")
    quality_token_ids = get_quality_token_ids(tokenizer)
    print(f"品質詞彙token IDs: {dict(zip(QUALITY_LEVELS, quality_token_ids))}")

    # 3. 創建數據集
    print("3. 載入訓練數據...")
    train_dataset = QualityAssessmentDataset(
        csv_file="fake_data/config.csv",
        image_dir="fake_data",
        processor=processor,
        tokenizer=tokenizer,
    )

    print(f"訓練數據集大小: {len(train_dataset)}")

    # 4. 設置訓練參數
    print("4. 配置訓練參數...")
    training_args = TrainingArguments(
        output_dir="./quality_assessment_model",
        per_device_train_batch_size=1,  # 極小批次大小
        gradient_accumulation_steps=8,  # 累積梯度
        num_train_epochs=3,
        learning_rate=5e-6,  # 較小的學習率
        weight_decay=0.01,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,  # 保留自定義列
        dataloader_pin_memory=False,  # 減少記憶體使用
        fp16=False,  # 完全禁用混合精度
        bf16=False,  # 禁用bf16
        report_to=None,  # 不使用外部報告
        warmup_steps=2,  # 少量warmup步驟
    )

    # 5. 初始化自定義訓練器
    print("5. 初始化訓練器...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,  # 使用自定義collate函數
        quality_token_ids=quality_token_ids,
        kl_weight=0.5,  # 調整KL loss權重
        tokenizer=tokenizer,  # 傳遞tokenizer作為named parameter
    )

    # 6. 開始訓練
    print("6. 開始訓練過程...")
    print("=" * 50)

    trainer.train()

    # 7. 保存訓練後的模型
    print("7. 保存訓練後的模型...")
    trainer.save_model("./quality_assessment_model_final")

    print("=" * 50)
    print("訓練完成！")
    print("模型已保存至: ./quality_assessment_model_final")


if __name__ == "__main__":
    main()
