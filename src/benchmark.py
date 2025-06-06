"""
Dataset Benchmark Tool
使用 OpenAI SDK 搭配 Ollama 測試數學問題解決能力
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import concurrent.futures
from dataclasses import dataclass, asdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 隱藏 httpx 和 openai 的詳細日誌
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

@dataclass
class BenchmarkResult:
    """單個測試結果的數據結構"""
    question_id: int
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    response_time: float
    model_name: str
    timestamp: str
    error: Optional[str] = None

class MathBenchmark:
    """基準測試"""
    
    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: str = "ollama"):
        """
        初始化基準測試
        
        Args:
            base_url: Ollama API 端點
            api_key: API 金鑰（對 Ollama 來說可以是任意值）
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.results: List[BenchmarkResult] = []
    
    def load_jsonl_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        從JSONL文件載入數據集
        
        Args:
            file_path: JSONL檔案路徑
            
        Returns:
            問題數據列表
        """
        dataset = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():  # 跳過空行
                        try:
                            data = json.loads(line.strip())
                            # 確保必要的字段存在
                            if 'question' in data and 'answer' in data:
                                # 添加ID如果不存在
                                if 'id' not in data:
                                    data['id'] = line_num
                                dataset.append(data)
                            else:
                                logger.warning(f"行 {line_num + 1}: 缺少必要字段 'question' 或 'answer'")
                        except json.JSONDecodeError as e:
                            logger.error(f"行 {line_num + 1}: JSON解析錯誤: {e}")
                            
            logger.info(f"成功載入 {len(dataset)} 個問題從 {file_path}")
            return dataset
            
        except FileNotFoundError:
            logger.error(f"找不到檔案: {file_path}")
            raise
        except Exception as e:
            logger.error(f"載入數據集時發生錯誤: {e}")
            raise
    
    def extract_answer(self, text: str) -> str:
        """
        從模型回應中提取最終答案
        
        Args:
            text: 模型的完整回應
            
        Returns:
            提取的答案字符串
        """
        # 尋找常見的答案標記
        answer_markers = [
            "answer is",
            "#### ",
            "####",
        ]
        
        text_lower = text.lower()
        
        # 尋找最後出現的答案標記
        last_pos = -1
        for marker in answer_markers:
            pos = text_lower.rfind(marker)
            if pos > last_pos:
                last_pos = pos
        
        if last_pos != -1:
            # 從標記後開始提取
            answer_part = text[last_pos:].split('\n')[0]
            # 移除標記文字
            for marker in answer_markers:
                answer_part = answer_part.lower().replace(marker, "").strip()
            return answer_part.strip(": ").strip()
        
        # 如果沒找到標記，返回最後一行
        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else text.strip()
    
    def is_answer_correct(self, predicted: str, expected: str) -> bool:
        """
        比較預測答案和期望答案是否一致
        
        Args:
            predicted: 模型預測的答案
            expected: 期望的正確答案
            
        Returns:
            是否正確
        """
        # 簡單的字符串匹配（可以根據需要改進）
        predicted_clean = predicted.lower().strip()
        expected_clean = expected.lower().strip()
        
        # 檢查完全匹配
        if predicted_clean == expected_clean:
            return True
            
        # 檢查是否包含期望答案
        if expected_clean in predicted_clean:
            return True
            
        # 數值匹配（嘗試解析為數字）
        try:
            pred_num = float(predicted_clean.replace(',', ''))
            exp_num = float(expected_clean.replace(',', ''))
            return abs(pred_num - exp_num) < 1e-6
        except:
            pass
            
        return False
    
    def test_single_question(self, question_data: Dict[str, Any], model_name: str) -> BenchmarkResult:
        """
        測試單個問題
        
        Args:
            question_data: 問題數據
            model_name: 模型名稱
            
        Returns:
            測試結果
        """
        question = question_data["question"]
        expected_answer = question_data["answer"]
        question_id = question_data.get("id", 0)
        
        prompt = f"""Please solve the following math problem and give the final answer clearly at the end.
problem: {question}
Please solve this problem step by step and clearly state your final answer at the "#### " end

Example:

problem:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
answer: 
Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
"""
        
        start_time = time.time()
        error = None
        model_answer = ""
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                seed=42,
                max_tokens=2048
            )
            
            model_answer = response.choices[0].message.content
            response_time = time.time() - start_time
            
            extracted_answer = self.extract_answer(model_answer)
            is_correct = self.is_answer_correct(extracted_answer, expected_answer)
            
        except Exception as e:
            response_time = time.time() - start_time
            error = str(e)
            is_correct = False
            logger.error(f"測試問題 {question_id} 時發生錯誤: {e}")
        
        return BenchmarkResult(
            question_id=question_id,
            question=question,
            expected_answer=expected_answer,
            model_answer=model_answer,
            is_correct=is_correct,
            response_time=response_time,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            error=error
        )
    
    def run_benchmark(self, model_name: str, input_file: str, 
                     max_workers: int = 4) -> List[BenchmarkResult]:
        """
        運行基準測試
        
        Args:
            model_name: 要測試的模型名稱
            input_file: JSONL輸入檔案路徑
            max_workers: 並行工作數量
            
        Returns:
            測試結果列表
        """
        logger.info(f"開始基準測試 - 模型: {model_name}")
        
        # 載入數據集
        dataset = self.load_jsonl_dataset(input_file)
        
        # 並行測試
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_question = {
                executor.submit(self.test_single_question, question_data, model_name): i
                for i, question_data in enumerate(dataset)
            }
            
            # 收集結果並顯示進度
            with tqdm(total=len(dataset), desc=f"測試 {model_name}") as pbar:
                for future in concurrent.futures.as_completed(future_to_question):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"處理結果時發生錯誤: {e}")
                        pbar.update(1)
        
        self.results.extend(results)
        logger.info(f"完成 {model_name} 的測試，共 {len(results)} 個樣本")
        return results
    
    def save_results(self, output_dir: str = "benchmark_results"):
        """
        保存測試結果到文件
        
        Args:
            output_dir: 輸出目錄
        """
        
        # 保存詳細結果為 JSON
        json_file = output_dir / f"detailed_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(result) for result in self.results], f, 
                     ensure_ascii=False, indent=2)
        
        # 保存匯總結果為 CSV
        df = pd.DataFrame([asdict(result) for result in self.results])
        csv_file = output_dir / f"summary_results.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"結果已保存到: {json_file} 和 {csv_file}")
        return json_file, csv_file
    
    def generate_charts(self, output_dir: str = "benchmark_results"):
        """
        生成分析圖表
        
        Args:
            output_dir: 輸出目錄
        """
        if not self.results:
            logger.warning("沒有結果可以生成圖表")
            return
        
        # 準備數據
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建子圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. 準確率比較
        accuracy_by_model = df.groupby('model_name')['is_correct'].mean()
        axes[0, 0].bar(accuracy_by_model.index, accuracy_by_model.values)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracy_by_model.values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. 回應時間分布
        sns.boxplot(data=df, x='model_name', y='response_time', ax=axes[0, 1])
        axes[0, 1].set_title('Response Time Distribution')
        axes[0, 1].set_ylabel('Response Time (seconds)')
        
        # 3. 準確率隨時間變化（如果有多個模型）
        if len(df['model_name'].unique()) > 1:
            for model in df['model_name'].unique():
                model_df = df[df['model_name'] == model].reset_index(drop=True)
                # 計算滑動平均
                window_size = max(1, len(model_df) // 20)
                rolling_acc = model_df['is_correct'].rolling(window=window_size, min_periods=1).mean()
                axes[1, 0].plot(rolling_acc.index, rolling_acc.values, label=model, marker='o', markersize=2)
            axes[1, 0].set_title('Accuracy Trend')
            axes[1, 0].set_xlabel('Question Index')
            axes[1, 0].set_ylabel('Rolling Accuracy')
            axes[1, 0].legend()
        else:
            # 單模型的情況，顯示回答正確性
            model_df = df.reset_index(drop=True)
            axes[1, 0].scatter(model_df.index, model_df['is_correct'], alpha=0.6)
            axes[1, 0].set_title('Answer Correctness by Question')
            axes[1, 0].set_xlabel('Question Index')
            axes[1, 0].set_ylabel('Correct (1) / Incorrect (0)')
        
        # 4. 統計摘要表
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        summary_stats = []
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            stats = {
                'Model': model,
                'Total Questions': len(model_df),
                'Correct': model_df['is_correct'].sum(),
                'Accuracy': f"{model_df['is_correct'].mean():.3f}",
                'Avg Time (s)': f"{model_df['response_time'].mean():.2f}",
                'Errors': model_df['error'].notna().sum()
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        table = axes[1, 1].table(cellText=summary_df.values,
                                colLabels=summary_df.columns,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # 保存圖表
        chart_file = output_dir / f"benchmark_charts.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        # plt.show()
        
        logger.info(f"圖表已保存到: {chart_file}")
        return chart_file




def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Dataset Benchmark Tool")
    parser.add_argument("--models", nargs="+", default=["llama3.2"], 
                       help="要測試的模型列表")
    parser.add_argument("--input-file", type=str, required=True,
                       help="輸入JSONL檔案路徑")
    parser.add_argument("--workers", type=int, default=4,
                       help="並行工作數量")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="輸出目錄")
    parser.add_argument("--base-url", default="http://localhost:11434/v1",
                       help="Ollama API 端點")
    
    args = parser.parse_args()


    # 創建基準測試實例
    benchmark = MathBenchmark(base_url=args.base_url)
    
    try:
        # 對每個模型運行測試
        for model_name in args.models:
            logger.info(f"開始測試模型: {model_name}")
            benchmark.run_benchmark(
                model_name=model_name,
                input_file=args.input_file,
                max_workers=args.workers
            )
        
        # 保存結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = Path(f"{args.output_dir}/{timestamp}")
        if not output_path.exists():
            logger.info(f"創建輸出目錄: {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)  

        # 保存詳細結果和摘要
        json_file, csv_file = benchmark.save_results(output_path)
        
        # 生成圖表
        chart_file = benchmark.generate_charts(output_path)
        
        # 打印摘要
        df = pd.DataFrame([asdict(result) for result in benchmark.results])
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            accuracy = model_df['is_correct'].mean()
            avg_time = model_df['response_time'].mean()
            total_questions = len(model_df)
            errors = model_df['error'].notna().sum()
            
            print(f"\n模型: {model}")
            print(f"  測試問題數: {total_questions}")
            print(f"  正確答案數: {model_df['is_correct'].sum()}")
            print(f"  準確率: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"  平均回應時間: {avg_time:.2f} 秒")
            print(f"  錯誤數: {errors}")
        
        print(f"\n結果已保存到:")
        print(f"  詳細結果: {json_file}")
        print(f"  摘要結果: {csv_file}")
        print(f"  分析圖表: {chart_file}")
        
    except KeyboardInterrupt:
        logger.info("測試被用戶中斷")
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main()
