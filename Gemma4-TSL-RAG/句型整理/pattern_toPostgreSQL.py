import json
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer

# 1. 初始化向量模型 (推薦使用適合中文的輕量化模型)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # 維度為 384

# 2. 資料庫連線設定
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
conn = psycopg2.connect(
    dbname="tsl_rag_system",
    user="postgres",
    password=os.getenv("DB_PASSWORD"),
    host="localhost",
    port="5432"
)
cur = conn.cursor()


def import_json_to_pg(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    book_name = data['教材名稱']
    
    for unit in data['單元內容']:
        u_name = unit['單元名稱']
        for pattern in unit['句型練習']:
            # 準備存入的資料
            p_no = pattern['句型數字']
            desc = pattern['句型功能描述']
            logic = pattern['邏輯']
            sentence = pattern['例句']
            tsl = pattern['TSL']
            
            # 將例句轉換為向量
            print(f"正在處理: {sentence}")
            embedding = model.encode(sentence).tolist()
            
            # 寫入資料庫
            cur.execute("""
                INSERT INTO tsl_knowledge 
                (book_name, unit_name, pattern_no, description, logic_structure, chinese_sentence, tsl_markup, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (book_name, u_name, p_no, desc, logic, sentence, tsl, embedding))
            
    conn.commit()
    print(f"{book_name} 匯入完成！")

# 3. 批次處理 9-18 冊 (假設檔名規律)
for i in range(9, 19):
    try:
        import_json_to_pg(DATA_DIR / f'B{i}.json')
    except FileNotFoundError:
        continue

cur.close()
conn.close()
