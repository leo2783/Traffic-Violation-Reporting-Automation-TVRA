"""
Traffic Violation Reporting Automation (TVRA) - Local Dataset Module
Responsible for scanning video or image files in a specified path.
本模組負責掃描指定路徑下的影片或圖片檔案。
"""

import os


try:
    from .interfaces import BaseDataset
except ImportError:
    from interfaces import BaseDataset
from typing import List

class LocalDataset(BaseDataset):
    """
    Local Folder Dataset Class
    Responsible for scanning video or image files in a specified path.
    
    本地資料夾資料集類別
    負責掃描指定路徑下的影片或圖片檔案
    """
    def __init__(self, directory: str):
        """
        Initialize LocalDataset
        :param directory: Path to the local folder containing video/image files
        """
        if not directory or not isinstance(directory, str):
            raise ValueError("directory 不能為空，且必須是字串")
        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise FileNotFoundError(f"找不到指定的目錄: {directory}")
            
        self._directory = directory
        
    def get_sources(self) -> List[str]:
        """
        Scan for supported formats in the folder.
        掃描資料夾內支援的格式。
        :return: List of file paths
        """
        supported_formats = ('.mp4', '.avi', '.mov', '.ts', '.jpg', '.jpeg', '.png')
        sources = []
        if os.path.exists(self._directory):
            for f in os.listdir(self._directory):
                if f.lower().endswith(supported_formats):
                    sources.append(os.path.join(self._directory, f))
        return sources

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(directory='{self._directory}')>"
