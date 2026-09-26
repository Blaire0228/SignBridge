import torch

def check_gpu():
    print(f"--- GPU 環境檢查 ---")
    
    # 1. 檢查 PyTorch 是否支援 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch 是否支援 CUDA: {cuda_available}")
    
    if cuda_available:
        # 2. 取得 GPU 數量與名稱
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"找到 GPU 數量: {device_count}")
        print(f"當前使用 GPU 名稱: {device_name}")
        
        # 3. 檢查顯存 (VRAM) 狀態
        # 轉換成 GB 比較好讀
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"顯卡總 VRAM 容量: {total_memory:.2f} GB")
        
        # 4. 簡單的矩陣運算測試 (真正把資料丟進顯卡跑跑看)
        try:
            x = torch.rand(100, 100).cuda()
            print("矩陣運算測試: 成功 (資料已進入 GPU)")
        except Exception as e:
            print(f"矩陣運算測試: 失敗 ({str(e)})")
            
    else:
        print("警告: 找不到可用的 NVIDIA GPU！請檢查顯卡驅動或 PyTorch 版本。")

if __name__ == "__main__":
    check_gpu()