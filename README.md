## Benchmark Tool - 安裝與使用指南

### 🛠️ 環境設定

1. **Python 3.8+**
   檢查版本：`python --version`

2. **建立虛擬環境（推薦）**

   ```bash
   uv venv

   # Linux/Mac
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

3. **安裝依賴**

   ```bash
   uv sync
   # or
   uv add -r requirements.txt
   ```

---

### 🚀 Ollama 設置

1. **安裝並啟動**：下載 Ollama → `ollama serve`
2. **下載模型**：

   ```bash
   ollama pull llama3.2
   ollama list
   ```

---

### 📊 基本用法

```bash
# 單模型測試
python src/benchmark.py --models llama3.2 --input-file data.jsonl

# 多模型 & 並行處理
python src/benchmark.py --models llama3.2 gemma2:2b --input-file data.jsonl --workers 4

# 自定義輸出與端點
python src/benchmark.py --models llama3.2 --input-file data.jsonl --output-dir ./results --base-url http://localhost:11434/v1
```

### 📝 JSONL 資料格式

輸入檔案必須是 JSONL 格式，每行包含一個 JSON 物件：
可以參考: `prepare_dataset.py`  

```jsonl
{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"}
{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"}
```

**必要欄位：**
- `question`: 數學問題文字
- `answer`: 正確答案
- `id`: 問題ID（選用，未提供時會自動產生）

---

### 🔧 常見問題

* **JSONL 格式錯誤**：確保每行都是有效的 JSON 物件，包含必要的 `question` 和 `answer` 欄位
* **檔案路徑錯誤**：檢查 `--input-file` 參數指向的檔案是否存在
* **模型異常**：

  ```bash
  ollama list
  ollama run llama3.2 "test"
  ollama rm llama3.2 && ollama pull llama3.2
  ```
* **資源不足**：減少 `--workers` 或使用較小模型

---

### 📈 輸出說明

* `detailed_results_*.json`: 問題詳情
* `summary_results_*.csv`: 摘要統計
* `benchmark_charts_*.png`: 圖表（準確率、反應時間等）
