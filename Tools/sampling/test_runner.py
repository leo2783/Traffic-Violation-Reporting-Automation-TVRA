"""
Traffic Violation Reporting Automation (TVRA) - Unified Test Runner
Supports testing YOLO inference on local videos, local images, YouTube streams, or single files.
統一測試工具，支援對本地影片、本地圖片、YouTube 串流或單一檔案進行 YOLO 推論測試。

Usage:
  # 測試本地影片資料夾
  python test_runner.py --source video --path ./test_video

  # 測試本地圖片資料夾
  python test_runner.py --source image --path ./test_images

  # 測試 YouTube 串流
  python test_runner.py --source youtube --count 5

  # 測試單一檔案
  python test_runner.py --source file --path ./test.mp4
"""

import argparse
import logging
import os
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def test_video_folder(model, folder_path, conf, output):
    """Test all videos in a local folder"""
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.ts'))]
    
    if not video_files:
        logger.warning(f"資料夾中沒有找到影片檔案: {folder_path}")
        return

    logger.info(f"找到 {len(video_files)} 部影片，開始處理...")
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(folder_path, video_file)
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"無法開啟影片: {video_file}")
            continue
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"正在處理 [{i+1}/{len(video_files)}]: {video_file} (原始尺寸: {width}x{height})")
        
        results = model.predict(
            source=video_path,
            conf=conf,
            save=True,
            show=False,
            stream=True,
            name=os.path.splitext(video_file)[0] if not output else os.path.join(output, os.path.splitext(video_file)[0]),
            exist_ok=True
        )
        
        for r in results:
            pass
    
    logger.info(f"所有影片處理完成！")


def test_image_folder(model, folder_path, conf, output):
    """Test all images in a local folder"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        logger.warning(f"資料夾中沒有找到圖片檔案: {folder_path}")
        return

    logger.info(f"找到 {len(image_files)} 張圖片，開始處理...")
    
    results = model.predict(
        source=folder_path,
        conf=conf,
        save=True,
        show=False,
        name="test_images" if not output else os.path.join(output, "test_images"),
        exist_ok=True
    )
    
    logger.info(f"圖片處理完成！")


def test_single_file(model, file_path, conf, output):
    """Test a single video or image file"""
    if not os.path.exists(file_path):
        logger.error(f"檔案不存在: {file_path}")
        return
    
    file_name = os.path.basename(file_path)
    logger.info(f"正在處理單一檔案: {file_name}")
    
    results = model.predict(
        source=file_path,
        conf=conf,
        save=True,
        show=False,
        stream=file_path.lower().endswith(('.mp4', '.avi', '.mov', '.ts')),
        name=os.path.splitext(file_name)[0] if not output else os.path.join(output, os.path.splitext(file_name)[0]),
        exist_ok=True
    )
    
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.ts')):
        for r in results:
            pass
    
    logger.info(f"檔案處理完成！")


def get_youtube_stream_url(source_url: str) -> str:
    """取得 YouTube 影片的直接串流網址"""
    import yt_dlp
    ydl_opts = {'format': 'best', 'quiet': True, 'no_warnings': True, 'noplaylist': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source_url, download=False)
        return info.get('url', source_url)

def test_youtube(model, count, conf, output):
    """Test YouTube streams by scraping video URLs"""
    try:
        from youtube_dataset import YoutubeDataset
    except ImportError:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sampling"))
        from youtube_dataset import YoutubeDataset
    
    dataset = YoutubeDataset(target_count=count)
    sources = dataset.get_sources()
    
    logger.info(f"取得 {len(sources)} 個 YouTube 影片，開始處理...")
    
    for i, source in enumerate(sources):
        logger.info(f"正在處理 YouTube [{i+1}/{len(sources)}]: {source}")
        try:
            video_url = get_youtube_stream_url(source)
            
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                logger.warning(f"無法開啟 YouTube 串流: {source}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            cap.release()
            
            results = model.predict(
                source=video_url,
                conf=conf,
                save=True,
                show=False,
                stream=True,
                name=f"youtube_{i}" if not output else os.path.join(output, f"youtube_{i}"),
                exist_ok=True
            )
            
            for r in results:
                pass
                
        except Exception as e:
            logger.warning(f"YouTube 處理失敗: {e}")
    
    logger.info("YouTube 測試完成！")


def main():
    parser = argparse.ArgumentParser(description="TVRA 統一測試工具")
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
    
    # 設定全域 Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 載入模型
    logger.info(f"正在載入 YOLO 模型: {args.yolo_weights}")
    model = YOLO(args.yolo_weights)
    
    # 根據來源類型執行對應測試
    if args.source == "video":
        if not args.path:
            logger.error("source=video 時需要指定 --path")
            return
        test_video_folder(model, args.path, args.conf, args.output)
    
    elif args.source == "image":
        if not args.path:
            logger.error("source=image 時需要指定 --path")
            return
        test_image_folder(model, args.path, args.conf, args.output)
    
    elif args.source == "file":
        if not args.path:
            logger.error("source=file 時需要指定 --path")
            return
        test_single_file(model, args.path, args.conf, args.output)
    
    elif args.source == "youtube":
        test_youtube(model, args.count, args.conf, args.output)


if __name__ == "__main__":
    main()