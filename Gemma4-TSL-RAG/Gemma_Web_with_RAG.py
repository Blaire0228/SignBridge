import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import time
from psycopg2.pool import ThreadedConnectionPool
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from copy import deepcopy
from threading import Lock

# 加載環境變數
load_dotenv()

DB_POOL_MIN_CONNECTIONS = int(os.getenv("DB_POOL_MIN_CONNECTIONS", "1"))
DB_POOL_MAX_CONNECTIONS = int(os.getenv("DB_POOL_MAX_CONNECTIONS", "4"))
_db_pool = None
_db_pool_lock = Lock()

def initialize_db_pool():
    """Create the process-wide PostgreSQL connection pool once."""
    global _db_pool
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                _db_pool = ThreadedConnectionPool(
                    minconn=DB_POOL_MIN_CONNECTIONS,
                    maxconn=DB_POOL_MAX_CONNECTIONS,
                    dbname="tsl_rag_system",
                    user="postgres",
                    password=os.getenv("DB_PASSWORD"),
                    host="localhost",
                )
    return _db_pool


def close_db_pool():
    """Close every pooled PostgreSQL connection during shutdown."""
    global _db_pool
    with _db_pool_lock:
        if _db_pool is not None:
            _db_pool.closeall()
            _db_pool = None

# ================= 1. 系統初始化（直接 model.generate + 4-bit 量化）=================
MODEL_ID = "google/gemma-4-E4B-it"
HF_TOKEN = os.getenv("HF_TOKEN")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"正在載入模型核心: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    clean_up_tokenization_spaces=False,
)
raw_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0}, # 強制全 GPU 運行，防止 Offload 報錯
    token=HF_TOKEN,
    low_cpu_mem_usage=True
)

print("模型載入完成。")

generation_config = deepcopy(raw_model.generation_config)
generation_config.max_length = None
generation_config.max_new_tokens = 32
generation_config.do_sample = False
generation_config.use_cache = True
generation_config.eos_token_id = [
    tokenizer.eos_token_id,
    tokenizer.eot_token_id,
]
generation_config.pad_token_id = tokenizer.pad_token_id

embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

# ================= 2. 核心功能函數 (雙軌檢索與 Prompt 增強) =================

def get_rag_context(query_text):
    """分別從兩表定額抽樣，確保必有教科書句型邏輯"""
    started = time.perf_counter()
    query_vector = embed_model.encode(query_text).tolist()
    embedding_seconds = time.perf_counter() - started

    query_started = time.perf_counter()
    pool = initialize_db_pool()
    conn = pool.getconn()
    
    # 一次查詢取得教科書範例與生活語料，減少資料庫往返。
    sql_combined = """
        (SELECT
            chinese_sentence,
            tsl_markup AS tsl,
            logic_structure AS logic,
            1 - (embedding <=> %s::vector) AS sim
        FROM tsl_knowledge
        ORDER BY sim DESC
        LIMIT 2)
        UNION ALL
        (SELECT
            chinese_text AS chinese_sentence,
            tsl_text AS tsl,
            '一般生活對話' AS logic,
            1 - (embedding <=> %s::vector) AS sim
        FROM tsl_general_corpus
        ORDER BY sim DESC
        LIMIT 1);
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql_combined, (query_vector, query_vector))
            all_rows = cur.fetchall()
    except Exception:
        if not conn.closed:
            conn.rollback()
        pool.putconn(conn, close=bool(conn.closed))
        raise
    else:
        conn.rollback()
        pool.putconn(conn)

    query_seconds = time.perf_counter() - query_started
    print(
        "[Gemma timing] "
        f"embedding={embedding_seconds:.3f}s "
        f"database={query_seconds:.3f}s",
        flush=True,
    )
    return all_rows

def translate_tsl(chinese_input):
    """依照官方範本修改的標準對話流"""
    contexts = get_rag_context(chinese_input)
    
    # 1. 精簡但保留既有語序規則與輸出限制的 System Instruction
    system_instruction = (
        "你是台灣手語（TSL）翻譯器。請遵守：\n"
        "1. 時間、地點置於句首。\n"
        "2. 主體不可省略，置於時間／地點之後、受詞之前。\n"
        "3. 受詞置於動詞之前。\n"
        "4. 意願助動詞（想、喜歡、要）置於主要動作之後。\n\n"
        "語序：時間／地點 + 主體 + 受詞 + 動詞 + 意願\n"
        "例：「他今天晚上想看電視」→「今天 晚上 他 電視 看 想」\n\n"
        "只輸出以空白分隔的 TSL 標記，不得包含解釋、中文原句、標點、控制標籤或其他文字。"
    )
    
    # 2. User Content：塞入外部 RAG 知識庫與最終任務
    user_content = "參考例：\n"
    context_str = ""
    for i, (chn, tsl, logic, _sim) in enumerate(contexts, 1):
        ref_block = (
            f"{i}. 中文：{chn}\n"
            f"   語法：{logic}\n"
            f"   TSL：{tsl}\n"
        )
        user_content += ref_block
        context_str += ref_block
    
    user_content += (
        f"\n翻譯：{chinese_input}\n"
        f"TSL："
    )

    # 3. 建立標準 messages 陣列
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]
    
    display_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(raw_model.device)

    with torch.inference_mode():
        output_ids = raw_model.generate(
            **model_inputs,
            generation_config=generation_config,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_length:]

    tsl_result = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    tsl_result = tsl_result.replace("`", "").replace("\n", " ").strip()

    return tsl_result, display_prompt, context_str


def reconstruct_sentence(words: list[str]) -> str:
    """Convert an ordered TSL gloss sequence into natural Traditional Chinese."""
    cleaned_words = [word.strip() for word in words if word and word.strip()]
    if not cleaned_words:
        return ""
    if len(cleaned_words) == 1:
        return cleaned_words[0]

    gloss_text = " ".join(cleaned_words)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一位專業的台灣手語（TSL）語意解析專家。使用者會提供依序辨識出的"
                "台灣手語詞彙，可能包含倒裝、省略連接詞或受詞前置。請將這些詞彙重組為"
                "通順自然的繁體中文句子。可以補上必要的助詞、介系詞或連接詞，但必須忠於"
                "原意，不得加入輸入未表達的人物、地點、時間或事件。只輸出一行包含適當"
                "標點的繁體中文句子，不要提供解釋或程式碼區塊。"
            ),
        },
        {"role": "user", "content": f"手語詞彙：{gloss_text}"},
    ]
    config = deepcopy(generation_config)
    config.max_new_tokens = 32
    model_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(raw_model.device)

    with torch.inference_mode():
        output_ids = raw_model.generate(
            **model_inputs,
            generation_config=config,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prompt_length = model_inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_length:]

    output = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    output = output.replace("`", "").replace("\n", " ").strip()
    return output or gloss_text


def warm_up_model():
    """Initialize CUDA and generation kernels before the first real request."""
    warmup_config = deepcopy(generation_config)
    warmup_config.max_new_tokens = 1
    started = time.perf_counter()
    messages = [
        {"role": "user", "content": "測試"},
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(raw_model.device)

    with torch.inference_mode():
        raw_model.generate(
            **model_inputs,
            generation_config=warmup_config,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        f"[Gemma startup] model warm-up={time.perf_counter() - started:.3f}s",
        flush=True,
    )
