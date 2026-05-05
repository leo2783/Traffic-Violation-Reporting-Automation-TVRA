"""
Traffic Violation Reporting Automation (TVRA) - Unified Test Runner (Refactored)
Supports testing YOLO inference on local videos, local images, YouTube streams, or single files.
統一測試工具，支援對本地影片、本地圖片、YouTube 串流或單一檔案進行 YOLO 推論測試。
"""

import argparse
import logging
import os
import cv2
from pathlib import Path

from utils import YoloAnalyzer, path_check

logger = logging.getLogger(__name__)

class UnifiedTestRunner:
    """
    統一測試運行類別
    封裝了針對不同來源（影片、圖片、YouTube、檔案）的測試邏輯，並統一使用 YoloAnalyzer。
    """
    def __init__(self, yolo_weights: str, conf: float = 0.7):
        self._analyzer = YoloAnalyzer(yolo_weights)
        self._conf = conf

    def test_video_folder(self, folder_path: str, output: str = None) -> None:
        """測試本地影片資料夾"""
        folder = Path(folder_path)
        video_extensions = ('.mp4', '.avi', '.mov', '.ts')
        video_files = [f for f in folder.iterdir() if f.suffix.lower() in video_extensions]
        
        if not video_files:
            logger.warning(f"資料夾中沒有找到影片檔案: {folder_path}")
            return

        logger.info(f"找到 {len(video_files)} 部影片，開始處理...")
        
        for i, video_file in enumerate(video_files):
            logger.info(f"正在處理 [{i+1}/{len(video_files)}]: {video_file.name}")
            
            # 使用 YoloAnalyzer 進行推論
            # 注意：YoloAnalyzer.predict 內建了 stream=True
            results = self._analyzer.predict(
                source=str(video_file),
                conf=self._conf,
                save=True,
                show=False,
                name=video_file.stem if not output else os.path.join(output, video_file.stem),
                exist_ok=True
            )
            
            # 遍歷 generator 以觸發推論
            for _ in results:
                pass
        
        logger.info(f"所有影片處理完成！")

    def test_image_folder(self, folder_path: str, output: str = None) -> None:
        """測試本地圖片資料夾"""
        if not os.path.isdir(folder_path):
            logger.error(f"無效的圖片資料夾路徑: {folder_path}")
            return

        logger.info(f"開始處理圖片資料夾: {folder_path}")
        
        self._analyzer.predict(
            source=folder_path,
            conf=self._conf,
            save=True,
            show=False,
            name="test_images" if not output else os.path.join(output, "test_images"),
            exist_ok=True
        )
        
        logger.info(f"圖片處理完成！")

    def test_single_file(self, file_path: str, output: str = None) -> None:
        """測試單一檔案"""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"檔案不存在: {file_path}")
            return
        
        logger.info(f"正在處理單一檔案: {path.name}")
        
        is_video = path.suffix.lower() in ('.mp4', '.avi', '.mov', '.ts')
        
        results = self._analyzer.predict(
            source=str(path),
            conf=self._conf,
            save=True,
            show=False,
            name=path.stem if not output else os.path.join(output, path.stem),
            exist_ok=True
        )
        
        if is_video:
            for _ in results:
                pass
        
        logger.info(f"檔案處理完成！")

    def _get_youtube_stream_url(self, source_url: str) -> str:
        """取得 YouTube 影片的直接串流網址"""
        import yt_dlp
        ydl_opts = {'format': 'best', 'quiet': True, 'no_warnings': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_url, download=False)
            return info.get('url', source_url)

    def test_youtube(self, count: int, output: str = None) -> None:
        """測試 YouTube 串流"""
        try:
            from youtube_dataset import YoutubeDataset
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from youtube_dataset import YoutubeDataset
        
        dataset = YoutubeDataset(target_count=count)
        sources = dataset.get_sources()
        
        logger.info(f"取得 {len(sources)} 個 YouTube 影片，開始處理...")
        
        for i, source in enumerate(sources):
            logger.info(f"正在處理 YouTube [{i+1}/{len(sources)}]: {source}")
            try:
                video_url = self._get_youtube_stream_url(source)
                
                results = self._analyzer.predict(
                    source=video_url,
                    conf=self._conf,
                    save=True,
                    show=False,
                    name=f"youtube_{i}" if not output else os.path.join(output, f"youtube_{i}"),
                    exist_ok=True
                )
                
                for _ in results:
                    pass
                    
            except Exception as e:
                logger.warning(f"YouTube 處理失敗: {e}")
        
        logger.info("YouTube 測試完成！")

def main():
    parser = argparse.ArgumentParser(description="TVRA 統一測試工具 (Refactored)")
    parser.add_argument("--source", type=str, required=True,
                        choices=["video", "image", "youtube", "file"],
                        help="測試來源類型 (video/image/youtube/file)")
    parser.add_argument("--path", type=str, default=None,
                        help="本地資料夾或檔案路徑 (source=video/image/file 時使用)")
    parser.add_argument("--count", type=int, default=5,
                        help="YouTube 測試數量 (source=youtube 時使用，預設 5)")
    parser.add_argument("--yolo_weights", type=str, required=True,
                        help="YOLO 模型權重路徑")
    parser.add_argument("--conf", type=float, default=0.7,
                        help="信心度門檻 (預設 0.7)")
    parser.add_argument("--output", type=str, default=None,
                        help="輸出目錄 (預設由 YOLO 自動產生)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    runner = UnifiedTestRunner(yolo_weights=args.yolo_weights, conf=args.conf)
    
    if args.source == "video":
        if not args.path:
            logger.error("source=video 時需要指定 --path")
        else:
            runner.test_video_folder(args.path, args.output)
    
    elif args.source == "image":
        if not args.path:
            logger.error("source=image 時需要指定 --path")
        else:
            runner.test_image_folder(args.path, args.output)
    
    elif args.source == "file":
        if not args.path:
            logger.error("source=file 時需要指定 --path")
        else:
            runner.test_single_file(args.path, args.output)
    
    elif args.source == "youtube":
        runner.test_youtube(args.count, args.output)

if __name__ == "__main__":
    main()
