# SignBridge

<p align="center">
  <b>English</b> | <a href="README_zh.md">繁體中文</a>
</p>

<p align="center">
  <img src="docs/assets/signbridge-cover.png" alt="SignBridge: TSL Two-Way Real-Time Translation System" width="100%">
</p>

SignBridge is a two-way communication application built around Taiwanese Sign Language (TSL). It combines a Flutter mobile app, word-level sign recognition, Gemma-powered retrieval-augmented generation (RAG), and 3D sign animations to help bridge conversations between hearing and Deaf or hard-of-hearing people.

## Features

- **Text to sign:** Converts Traditional Chinese or multilingual input into a TSL gloss sequence and plays the corresponding animations on a 3D avatar.
- **Sign to text:** Recognizes individual signs through a mobile camera and lets the user review the detected words before composing a sentence.
- **AI word-order conversion:** Uses Gemma to turn an ordered sequence of confirmed sign words into natural Traditional Chinese.
- **Chat rooms:** Creates or joins real-time chat rooms for exchanging text, recognition results, and sign animations.
- **Speech assistance:** Supports speech input and text-to-speech output.
- **Multilingual input:** Detects non-Chinese input and translates it into Traditional Chinese before producing TSL output.

## Architecture

```text
Flutter App
   │
   ▼
Main API (FastAPI, port 8000)
   ├── Word-level sign recognition (MediaPipe + ST-GCN)
   ├── Conversation rooms and WebSocket communication
   ├── GLB animation selection, merging, and caching
   └── Gemma API client
             │
             ▼
      Gemma RAG API (port 8001)
         ├── Google Gemma 4
         ├── Sentence Transformers
         └── PostgreSQL + pgvector
```

## Technology Stack

- Mobile application: Flutter / Dart
- Main backend: Python 3.10, FastAPI, MediaPipe, PyTorch, OpenCV
- Sign recognition: Spatial-Temporal Graph Convolutional Network (ST-GCN)
- AI translation: Google Gemma 4, Transformers, BitsAndBytes
- Retrieval-augmented generation: Sentence Transformers, PostgreSQL, pgvector
- Animation: glTF / GLB, `pygltflib`
- Real-time communication: WebSocket

## Repository Structure

```text
SignBridge/
├── frontend/                             # Flutter mobile application
├── sign_recognition/                     # Sign-recognition service adapter
├── TSL-translator-real-time-recognition/ # ST-GCN model and feature pipeline
├── Gemma4-TSL-RAG/                       # Gemma translation and RAG API
├── adjusted_actions/                     # GLB sign-animation assets
├── tests/                                # Backend and WebSocket tests
├── main.py                               # Main FastAPI service
├── merge_glb.py                          # GLB animation merger
├── requirements.txt                     # Main backend dependencies
├── start-backend.ps1                    # Windows startup script
└── stop-backend.ps1                     # Windows shutdown script
```

## Open Source Model Releases & Compliance

SignBridge provides open-source models and datasets on the Hugging Face Hub under permissive licenses, ensuring transparency and reproducibility.

### Hugging Face Assets
- **Word-Level TSL Recognition Model (ST-GCN):**
  - Repository: [`dYang1/SignBridge-TSL-STGCN`](https://huggingface.co/dYang1/SignBridge-TSL-STGCN)
  - License: Apache License 2.0
  - Contents: Trained checkpoint (`best_model.pt`), network architecture (`model.py`), graph topology (`graph.py`), and landmark extraction modules.
- **Taiwanese Sign Language Landmark Dataset:**
  - Repository: [`dYang1/SignBridge-TSL-Landmarks`](https://huggingface.co/dYang1/SignBridge-TSL-Landmarks)
  - License: Creative Commons Attribution 4.0 International (CC-BY-4.0)
  - Contents: 55-keypoint skeletal time-series features and TSL gloss annotations.

### Provenance & Compliance Declaration
1. **Autonomous Development:** The ST-GCN model architecture, training routines, and pre-processing pipelines were engineered independently. No proprietary or pre-trained models from PRC-based entities or restricted foreign-adversary vendors are utilized.
2. **Localization & Data Integrity:** Training corpora were captured and annotated specifically for Taiwanese Sign Language (TSL) lexical and grammatical standards in Taiwan.
3. **Open Standards:** Released under Apache-2.0 (code & weights) and CC-BY-4.0 (datasets) to fulfill all open-source competition and security verification requirements.

## Requirements

- Windows 10 or 11
- Python 3.10 for the main backend
- Python 3.12 for the Gemma RAG service
- Flutter SDK 3.x
- PostgreSQL with the pgvector extension
- An NVIDIA CUDA-capable GPU for 4-bit Gemma inference
- An Android phone or Android Emulator
- A Hugging Face account with access to the configured Gemma model

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/Blaire0228/SignBridge.git
cd SignBridge
```

### 2. Create the main backend environment

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create the Gemma RAG environment

```powershell
cd Gemma4-TSL-RAG
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

The correct PyTorch package depends on your GPU and CUDA driver. If the default installation does not enable GPU acceleration, follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Configure environment variables

```powershell
Copy-Item .\Gemma4-TSL-RAG\.env.example .\Gemma4-TSL-RAG\.env
```

Edit `Gemma4-TSL-RAG/.env`:

```dotenv
HF_TOKEN=your_hugging_face_token
DB_PASSWORD=your_postgresql_password
```

### 5. Configure PostgreSQL

Create the database:

```sql
CREATE DATABASE tsl_rag_system;
```

Connect to `tsl_rag_system`, then create the extension and tables:

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

### 6. Install Flutter dependencies

```powershell
cd frontend
flutter pub get
cd ..
```

## Running the Backend

Make sure PostgreSQL is running and both Python environments exist, then run:

```powershell
.\start-backend.ps1
```

Default service addresses:

- Main API: `http://127.0.0.1:8000`
- Main API documentation: `http://127.0.0.1:8000/docs`
- Gemma RAG API: `http://127.0.0.1:8001`
- Gemma RAG API documentation: `http://127.0.0.1:8001/docs`

Stop the services with:

```powershell
.\stop-backend.ps1
```

## Running the Flutter App

### Android Emulator

The Android Emulator can reach the host through the default `http://10.0.2.2:8000` address:

```powershell
cd frontend
flutter run
```

### Android device over USB

Enable Developer Options and USB debugging, then run:

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" reverse tcp:8000 tcp:8000
cd frontend
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

### Android device on the same Wi-Fi network

Allow TCP port 8000 through Windows Firewall and replace `<YOUR_PC_LAN_IP>` with the backend computer's LAN address:

```powershell
cd frontend
flutter run --dart-define=API_BASE_URL=http://<YOUR_PC_LAN_IP>:8000
```

Do not hard-code temporary tunnel URLs or personal network addresses. Supply remote endpoints at build time through `API_BASE_URL`.

## License and Third-Party Materials

The project source code is licensed under the Apache License 2.0.

Gemma usage and redistribution are subject to the Google Gemma Terms of Use.

## Citations & References

If you build upon SignBridge or use its components, please cite the underlying models:

```bibtex
@misc{gemma20264,
  title={Gemma 4: Open Multimodal Language Models},
  author={Gemma Team},
  year={2026},
  publisher={Google DeepMind},
  howpublished={\url{https://ai.google.dev/gemma/docs/core/model_card_4}}
}

@software{SignBridge_TSL_STGCN_2026,
  author = {SignBridge Development Team},
  title = {SignBridge-TSL-STGCN: Taiwanese Sign Language Recognition via ST-GCN},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/dYang1/SignBridge-TSL-STGCN}}
}
```

## Authors and Contributors

- [@Blaire0228](https://github.com/Blaire0228)
- [@defyingYang](https://github.com/defyingYang)
- [@Tobermory0927](https://github.com/Tobermory0927)
- [@CHIEH1111](https://github.com/CHIEH1111)
- [@xuanx0701](https://github.com/xuanx0701)
