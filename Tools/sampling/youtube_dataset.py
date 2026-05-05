"""
Traffic Violation Reporting Automation (TVRA) - YouTube Dataset Module
Responsible for scraping video URLs from a specified channel using Selenium.
本模組負責使用 Selenium 爬取指定頻道的影片網址。
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


from interfaces import BaseDataset
from typing import List

class YoutubeDataset(BaseDataset):
    """
    YouTube Video Dataset Class
    Responsible for scraping video URLs from a specified channel using Selenium.
    
    YouTube 影片資料集類別
    負責使用 Selenium 爬取指定頻道的影片網址
    """
    def __init__(self, target_count: int = 200, channel_url: str = 'https://www.youtube.com/@WoWtchout/videos'):
        """
        Initialize YoutubeDataset
        :param target_count: Number of video URLs to collect (default 200)
        :param channel_url: YouTube channel videos page URL
        """
        if not isinstance(target_count, int) or target_count <= 0:
            raise ValueError("target_count 必須是大於 0 的整數")
        if not channel_url or not isinstance(channel_url, str):
            raise ValueError("channel_url 不能為空，且必須是字串")
            
        self._target_count = target_count
        self._url = channel_url
        
    def get_sources(self) -> List[str]:
        """
        Execute the scraper to obtain video links.
        執行爬蟲獲取影片連結。
        :return: List of video URLs
        """
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # Uncomment below to enable headless mode
        # options.add_argument("--headless")
        
        driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
        try:
            driver.get(self._url)
            driver.maximize_window()
            
            # Wait for key elements to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, 'content'))
            )
            print("Starting to fetch YouTube video links... / 開始擷取 YouTube 影片連結...")
            video_links = set()
            
            # Scroll page until enough links are collected
            while len(video_links) < self._target_count:
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)
                links = driver.execute_script(
                    'return Array.from(document.querySelectorAll("a[href*=\\"/watch?v=\\"]")).map(a => a.href);'
                )
                for link in links:
                    clean_link = link.split('&')[0]
                    video_links.add(clean_link)
                    if len(video_links) >= self._target_count:
                        break
                print(f"Currently fetched: {len(video_links)} videos / 目前已收集: {len(video_links)} 部影片")
                
            return list(video_links)[:self._target_count]
        finally:
            driver.quit()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(target_count={self._target_count}, url='{self._url}')>"
