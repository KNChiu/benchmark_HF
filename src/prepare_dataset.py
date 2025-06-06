import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from datasets import load_dataset

# 設置 logger
logger = logging.getLogger(__name__)


def load_dataset_samples(num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    載入數據集並自動保存為 JSONL 格式

    Args:
        num_samples: 要測試的樣本數量，None 表示全部
        
    Returns:
        數據集樣本列表
    """
    logger.info("正在載入 數據集...")
    
    # 構建文件名
    samples_suffix = f"_{num_samples}" if num_samples else "_all"
    jsonl_path = Path(f"Dataset{samples_suffix}.jsonl")
    
    # 檢查文件是否已存在
    if jsonl_path.exists():
        logger.info(f"找到已存在的數據集文件 {jsonl_path}，正在載入...")
        try:
            processed_dataset = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed_dataset.append(json.loads(line))
            
            logger.info(f"已從緩存載入: {len(processed_dataset)} 個樣本")
            return processed_dataset
        except Exception as e:
            logger.warning(f"載入緩存文件失敗: {e}，將重新下載數據集")
    
    # 載入並處理數據集
    try:
        # 處理 split 參數
        if num_samples is None:
            split = "test"
        else:
            split = f"test[:{num_samples}]"
        
        logger.info(f"正在從 Hugging Face 下載數據集 (split: {split})...")
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        
        processed_dataset = []
        for item in dataset:
            # 提取最終答案（#### 後的數字）
            answer_parts = item['answer'].split('#### ')
            if len(answer_parts) >= 2:
                final_answer = answer_parts[-1].strip()
            else:
                # 如果沒有 #### 分隔符，嘗試提取最後的數字
                final_answer = item['answer'].strip()
            
            processed_item = {
                'question': item['question'].strip(),
                'answer': final_answer,
                'original_answer': item['answer']  # 保留原始答案以供參考
            }
            processed_dataset.append(processed_item)

        # 保存為 JSONL 格式
        logger.info(f"正在保存數據集到 {jsonl_path}...")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)  # 確保目錄存在
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in processed_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"已載入並保存: {len(processed_dataset)} 個樣本")
        
        return processed_dataset
    
    except Exception as e:
        logger.error(f"載入數據集時發生錯誤: {e}")
        raise


def validate_dataset_format(dataset: List[Dict[str, Any]]) -> bool:
    """
    驗證數據集格式是否正確
    
    Args:
        dataset: 數據集樣本列表
        
    Returns:
        格式是否有效
    """
    if not dataset:
        logger.error("數據集為空")
        return False
    
    required_keys = {'question', 'answer'}
    
    for i, item in enumerate(dataset):
        if not isinstance(item, dict):
            logger.error(f"第 {i} 個樣本不是字典格式")
            return False
        
        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            logger.error(f"第 {i} 個樣本缺少必要的鍵: {missing_keys}")
            return False
        
        if not item['question'] or not item['answer']:
            logger.error(f"第 {i} 個樣本的問題或答案為空")
            return False
    
    logger.info(f"數據集格式驗證通過，共 {len(dataset)} 個樣本")
    return True


def get_dataset_info(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    獲取數據集統計信息
    
    Args:
        dataset: 數據集樣本列表
        
    Returns:
        數據集統計信息
    """
    if not dataset:
        return {}
    
    # 統計問題長度
    question_lengths = [len(item['question']) for item in dataset]
    answer_lengths = [len(item['answer']) for item in dataset]
    
    info = {
        'total_samples': len(dataset),
        'question_stats': {
            'min_length': min(question_lengths),
            'max_length': max(question_lengths),
            'avg_length': sum(question_lengths) / len(question_lengths)
        },
        'answer_stats': {
            'min_length': min(answer_lengths),
            'max_length': max(answer_lengths),
            'avg_length': sum(answer_lengths) / len(answer_lengths)
        }
    }
    
    return info


if __name__ == "__main__":
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 測試載入小樣本
    try:
        samples = load_dataset_samples(num_samples=100)
        
        # 驗證格式
        if validate_dataset_format(samples):
            # 顯示統計信息
            info = get_dataset_info(samples)
            print(f"數據集信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
            
            # 顯示前幾個樣本
            print("\n前 3 個樣本:")
            for i, sample in enumerate(samples[:3]):
                print(f"\n樣本 {i+1}:")
                print(f"問題: {sample['question']}")
                print(f"答案: {sample['answer']}")
        
    except Exception as e:
        logger.error(f"測試運行失敗: {e}")
