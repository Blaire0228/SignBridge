[English](README.en.md) | **繁體中文**

# SignBridge

<p align="center">
  <img src="docs/assets/signbridge-cover.png" alt="SignBridge：台灣手語雙向即時翻譯系統" width="100%">
</p>

SignBridge 是一套以台灣手語（Taiwanese Sign Language, TSL）為核心的雙向溝通應用程式，整合 Flutter 行動端、手語單詞辨識、Gemma RAG 翻譯與 3D 手語動畫，協助聽人與聾人／聽障者進行即時交流。

## 主要功能

- **文字轉手語**：將繁體中文或其他語言輸入轉換為台灣手語詞序，並以 3D 角色播放手語動畫。
- **手語轉文字**：透過手機相機逐詞辨識手語，再由使用者確認並組成句子。
- **AI 語序轉換**：使用 Gemma 將已確認的手語單詞序列整理成自然的繁體中文。
- **聊天室**：建立或加入即時對話房間，交換文字、辨識結果與動畫內容。
- **語音輔助**：支援語音輸入與文字轉語音，降低不同溝通方式之間的阻礙。
- **多語言輸入**：偵測非中文輸入並先轉換為繁體中文，再進行台灣手語翻譯。

## 系統架構

```text
Flutter App
   │
   ▼
Main API (FastAPI, port 8000)
   ├── 手語單詞辨識（MediaPipe + PyTorch）
   ├── 對話房間與 WebSocket
   ├── GLB 動畫選取、合併與快取
   └── 呼叫 Gemma API
             │
             ▼
      Gemma RAG API (port 8001)
         ├── Google Gemma 4
         ├── Sentence Transformers
         └── PostgreSQL + pgvector
```

## 技術組成

- 行動端：Flutter / Dart
- 主後端：Python 3.10、FastAPI、MediaPipe、PyTorch、OpenCV
- AI 翻譯：Google Gemma 4、Transformers、BitsAndBytes
- RAG：Sentence Transformers、PostgreSQL、pgvector
- 動畫：glTF / GLB、`pygltflib`
- 即時通訊：WebSocket

## 專案結構

```text
SignBridge/
├── frontend/                            # Flutter 行動端
├── sign_recognition/                    # 單詞辨識服務封裝
├── TSL-translator-real-time-recognition/ # 單詞辨識模型與特徵處理
├── Gemma4-TSL-RAG/                      # Gemma 翻譯與 RAG API
├── adjusted_actions/                    # 手語 GLB 動作資產
├── tests/                               # 後端與 WebSocket 測試
├── main.py                              # 主 FastAPI 服務
├── merge_glb.py                         # GLB 動畫合併工具
├── requirements.txt                    # 主後端 Python 相依套件
├── start-backend.ps1                   # Windows 後端啟動腳本
└── stop-backend.ps1                    # Windows 後端停止腳本
```

## 執行需求

- Windows 10 或 11
- Python 3.10（主後端）
- Python 3.12（Gemma RAG）
- Flutter SDK 3.x
- PostgreSQL 與 pgvector extension
- 支援 CUDA 的 NVIDIA GPU（Gemma 4-bit 推論使用）
- Android 手機或 Android Emulator
- 已取得 Gemma 模型存取權限的 Hugging Face 帳號

## 安裝

### 1. 下載專案

```powershell
git clone https://github.com/Blaire0228/SignBridge.git
cd SignBridge
```

### 2. 建立主後端環境

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 建立 Gemma RAG 環境

```powershell
cd Gemma4-TSL-RAG
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

PyTorch 與 CUDA 的組合會因顯示卡及驅動版本而異。如預設安裝未啟用 GPU，請依 [PyTorch 官方安裝方式](https://pytorch.org/get-started/locally/) 安裝符合本機 CUDA 的版本。

### 4. 設定環境變數

複製範例檔：

```powershell
Copy-Item .\Gemma4-TSL-RAG\.env.example .\Gemma4-TSL-RAG\.env
```

編輯 `Gemma4-TSL-RAG/.env`：

```dotenv
HF_TOKEN=your_hugging_face_token
DB_PASSWORD=your_postgresql_password
```

### 5. 建立 PostgreSQL 資料庫

建立資料庫並啟用 pgvector：

```sql
CREATE DATABASE tsl_rag_system;
```

連線到 `tsl_rag_system` 後執行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE tsl_knowledge (
    id bigserial PRIMARY KEY,
    book_name varchar(50),
    unit_name varchar(100),
    pattern_no integer,
    description text,
    logic_structure text,
    chinese_sentence text,
    tsl_markup text,
    embedding vector(384)
);

CREATE TABLE tsl_general_corpus (
    id bigserial PRIMARY KEY,
    chinese_text text,
    tsl_text text,
    embedding vector(384)
);
```

### 6. 安裝 Flutter 相依套件

```powershell
cd frontend
flutter pub get
cd ..
```

## 啟動後端

確認 PostgreSQL 正在執行，而且兩個 Python 虛擬環境皆已建立後：

```powershell
.\start-backend.ps1
```

服務預設位址：

- 主 API：`http://127.0.0.1:8000`
- 主 API 文件：`http://127.0.0.1:8000/docs`
- Gemma RAG API：`http://127.0.0.1:8001`
- Gemma RAG API 文件：`http://127.0.0.1:8001/docs`

停止服務：

```powershell
.\stop-backend.ps1
```

## 執行 Flutter App

### Android Emulator

Android Emulator 可使用預設的 `http://10.0.2.2:8000` 連線主機：

```powershell
cd frontend
flutter run
```

### USB 連接的 Android 手機

啟用手機的開發人員選項與 USB 偵錯後：

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" reverse tcp:8000 tcp:8000
cd frontend
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

### 同一個 Wi-Fi 網路

確認 Windows 防火牆允許 TCP 8000，並將 `<YOUR_PC_LAN_IP>` 換成執行後端電腦的區域網路 IP：

```powershell
cd frontend
flutter run --dart-define=API_BASE_URL=http://<YOUR_PC_LAN_IP>:8000
```

不要將臨時 tunnel URL 或個人網路位址寫入原始碼；需要遠端連線時，請在建置階段透過 `API_BASE_URL` 傳入。


## 授權與第三方內容

專案程式碼的授權會記載於根目錄 `LICENSE`。

Gemma 模型的使用及散布須遵守 Google Gemma Terms of Use。

## 作者與貢獻者

- [@Blaire0228](https://github.com/Blaire0228)
- [@defyingYang](https://github.com/defyingYang)
- [@Tobermory0927](https://github.com/Tobermory0927)
- [@CHIEH1111](https://github.com/CHIEH1111)
- [@xuanx0701](https://github.com/xuanx0701)
