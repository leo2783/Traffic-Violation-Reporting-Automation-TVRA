import os
import subprocess
import time
import math

def get_changed_files(folder_path="dataset"):
    """
    使用 git ls-files 取得資料夾內所有修改 (modified)、刪除 (deleted)、未追蹤 (others) 的檔案。
    使用 -z 參數避免特殊字元或空白導致路徑被加上引號。
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "-m", "-d", "-o", "--exclude-standard", folder_path],
            capture_output=True,
            check=True
        )
        # 依照 \0 分割並過濾空字串
        files = [f.decode('utf-8') for f in result.stdout.split(b'\0') if f]
        
        # 使用 set 去除可能重複的檔案狀態
        return list(set(files))
    except subprocess.CalledProcessError as e:
        print(f"❌ 取得變更檔案失敗: {e}")
        return []

def batch_push(folder_path="dataset", total_batches=15):
    files = get_changed_files(folder_path)
    if not files:
        print(f"🎉 找不到 {folder_path} 資料夾中的變更，沒有需要推送的內容。")
        return

    total_files = len(files)
    batch_size = math.ceil(total_files / total_batches)
    print(f"🚀 發現 {total_files} 個變更的檔案。")
    print(f"📦 準備分為 {total_batches} 個批次推送（每批約 {batch_size} 個檔案）...")

    # Windows command line limit is 32767 chars. We chunk git add to avoid exceeding it.
    ADD_CHUNK_SIZE = 200 

    for i in range(total_batches):
        # 取得這一批次要處理的所有檔案
        batch_files = files[i * batch_size : (i + 1) * batch_size]
        if not batch_files:
            break
            
        print(f"\n▶️ 正在處理第 {i + 1}/{total_batches} 批 ({len(batch_files)} 個檔案)...")
        
        try:
            # 1. 執行 git add (內部再分小塊，避免超過 Windows 命令列長度限制)
            for j in range(0, len(batch_files), ADD_CHUNK_SIZE):
                chunk = batch_files[j : j + ADD_CHUNK_SIZE]
                subprocess.run(["git", "add", "--all", "--"] + chunk, check=True)
            
            # 2. 執行 git commit
            # 先檢查是否有實際 staged 的變更 (避免 commit 失敗)
            status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if not status_res.stdout.strip():
                print(f"⚠️ 第 {i + 1} 批沒有實質變更，跳過 commit 與 push。")
                continue

            commit_msg = f"chore: update dataset batch {i + 1}/{total_batches}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            
            # 3. 執行 git push
            print("⏳ 正在推送到遠端儲存庫...")
            subprocess.run(["git", "push"], check=True)
            print(f"✅ 第 {i + 1} 批推送成功！")
            
            # 稍作暫停，避免頻繁 push 被伺服器阻擋
            if i < total_batches - 1:
                time.sleep(2)
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 處理第 {i + 1} 批時發生錯誤: {e}")
            print("請檢查網路連線或 Git 狀態後再試。")
            break

if __name__ == "__main__":
    batch_push("dataset", total_batches=15)
