import json
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

# 1. 初始化模型 (維持一致的 384 維模型)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. 連線資料庫
conn = psycopg2.connect(
    dbname="tsl_rag_system",
    user="postgres",
    password=os.getenv("DB_PASSWORD"),
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# 3. 讀取並匯入「資料集.json」
def import_general_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"開始匯入通用資料集，共 {len(data)} 筆...")
    
    for item in data:
        chinese = item['chinese_text']
        tsl = item['tsl']
        
        # 產生向量
        embedding = model.encode(chinese).tolist()
        
        # 寫入新表
        cur.execute("""
            INSERT INTO tsl_general_corpus (chinese_text, tsl_text, embedding)
            VALUES (%s, %s, %s)
        """, (chinese, tsl, embedding))
    
    conn.commit()
    print("通用資料集匯入完成！")

# 執行匯入 (請確認檔案路徑)
import_general_dataset(DATA_DIR / '資料集.json')

cur.close()
conn.close()
