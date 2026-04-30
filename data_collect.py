from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time 

options =webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
url=('https://www.youtube.com/@WoWtchout/videos')
driver = webdriver.Chrome(options=options,service=Service(ChromeDriverManager().install()))
driver.get(url)
driver.maximize_window()

element = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID,'content'))
)
print("Start fetching links...")

video_links = set()
TARGET_COUNT = 200

while len(video_links) < TARGET_COUNT:
    # 捲動到頁面底部
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(2)  # 等待新影片載入
    
    # 使用包含 /watch?v= 的 a 標籤抓取影片連結
    links = driver.execute_script(
        'return Array.from(document.querySelectorAll("a[href*=\\"/watch?v=\\"]")).map(a => a.href);'
    )
    
    for link in links:
        # 過濾掉多餘的參數，只保留純影片連結
        clean_link = link.split('&')[0]
        video_links.add(clean_link)
        
        if len(video_links) >= TARGET_COUNT:
            break
            
    print(f"Currently fetched: {len(video_links)} videos")

# 轉換為 list 並精確取前 TARGET_COUNT 個
video_links = list(video_links)[:TARGET_COUNT]

print(f"\n--- Successfully fetched {len(video_links)} videos ---")
for idx, link in enumerate(video_links, 1):
    print(f"{idx}: {link}")
