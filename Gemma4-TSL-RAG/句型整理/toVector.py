import psycopg2
from sentence_transformers import SentenceTransformer

# 1. 初始化向量模型 (搬到迴圈外，只讀取一次)
print("正在載入 Embedding 模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. 資料庫連線設定 (搬到迴圈外，維持連線)
conn = psycopg2.connect(
    dbname="tsl_rag_system",
    user="postgres",
    password="apa5208",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

while True:
    user_input = input("\n請輸入要查詢的句子 (或輸入 'exit' 離開)：")
    if user_input.lower() == 'exit':
        break

    # 3. 將輸入句子轉換為向量
    embedding = model.encode(user_input).tolist()

    print(embedding)  # 顯示向量內容，確認是否正確生成

    # 4. 直接在資料庫進行「向量相似度查詢」
    # 使用 <=> 計算餘弦距離，1 - 距離 = 相似度
    cur.execute("""
        SELECT chinese_sentence, tsl_markup, 1 - (embedding <=> %s::vector) AS similarity
        FROM tsl_knowledge
        ORDER BY similarity DESC
        LIMIT 5
    """, (embedding,))

    results = cur.fetchall()

    # 5. 顯示結果
    print(f"--- 針對「{user_input}」的 RAG 檢索結果 ---")
    for row in results:
        chinese_sentence, tsl_markup, similarity = row
        print(f"相似度: {similarity:.4f} | TSL標記: {tsl_markup} | 原句: {chinese_sentence}")

cur.close()
conn.close()