import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# 1. 簡化量化配置，移除會報錯的參數
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

MODEL_ID = "google/gemma-4-E4B-it"

print("正在初始化模型與 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. 使用更保險的載入方式
# 我們手動設定 max_memory 來確保顯卡不會爆掉
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    # 這裡很關鍵：強制把顯存限制在 4.5GB，讓剩下的自動流向 RAM
    max_memory={0: "4.5GiB", "cpu": "12GiB"}
)

# 3. 建立 Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 4. 範例與 Prompt 組裝 (保持不變)
examples = [
    {"ch": "你別再長篇大論,請長話短說。", "tsl": "你 長舌 上下文 不 說 簡潔"},
    {"ch": "電視一直宣導要小心詐騙集團。", "tsl": "電視 以前 廣告 屢次 騙 注意 要"},
    {"ch": "他花了200元簽樂透。", "tsl": "他 樂透 買 200元"},
    {"ch": "意外很幸運的中了參獎。", "tsl": "忽然 及格 第三名 好幸運"},
    {"ch": "她這麼漂亮竟然嫁給這個醜男人,真是令人意外。", "tsl": "女 臉 美 嫁 他 醜 意料之外"}
]

prompt = "<|turn>system\n你是一位台灣手語(TSL)翻譯專家。請參考範例將中文轉譯為精簡、具象的 TSL 標記。<turn|>\n"
prompt +="<|turn>user\n手語的基礎翻譯文法為主詞+受詞+動詞<turn|>\n"
for ex in examples:
    prompt += f"<|turn>user\n中文：{ex['ch']}<turn|>\n<|turn>model\nTSL：{ex['tsl']}<turn|>\n"

test_sentence = "警察最近一直在宣導過馬路要小心車子。"
prompt += f"<|turn>user\n現在請翻譯：\n中文：{test_sentence}<turn|>\n<|turn>model\n"

# 5. 執行推論
print("-" * 30)
print(f"測試句子：{test_sentence}")
print("正在生成 TSL 翻譯...")

outputs = pipe(
    prompt,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    stop_strings=["<turn|>"],
    tokenizer=tokenizer
)

print("\n[模型輸出結果]:")
print(outputs[0]['generated_text'].split("<|turn>model")[-1].strip())