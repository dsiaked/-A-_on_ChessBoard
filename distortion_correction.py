import argparse
import glob
import os

import cv2 as cv
import numpy as np


def load_calibration(calib_file):
    if not os.path.exists(calib_file):
        raise FileNotFoundError(
            f"캘리브레이션 파일이 없습니다: {calib_file}\n"
            "먼저 camera_calibration.py를 실행해 calibration_result.npz를 생성하세요."
        )
    data = np.load(calib_file, allow_pickle=True)
    required_keys = ["K", "dist_coeff"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            "캘리브레이션 파일에 필수 항목이 없습니다: "
            + ", ".join(missing)
            + "\n필수: K, dist_coeff"
        )

    K = data["K"]
    dist_coeff = data["dist_coeff"]
    return K, dist_coeff


def undistort_frame(frame, K, dist_coeff, map_cache):
    h, w = frame.shape[:2]
    if map_cache.get("size") != (w, h):
        map1, map2 = cv.initUndistortRectifyMap(
            K, dist_coeff, None, None, (w, h), cv.CV_32FC1
        )
        map_cache["size"] = (w, h)
        map_cache["map1"] = map1
        map_cache["map2"] = map2
    return cv.remap(frame, map_cache["map1"], map_cache["map2"], interpolation=cv.INTER_LINEAR)


def stack_compare_view(original, rectified):
    left = original.copy()
    right = rectified.copy()
    cv.putText(left, "Original", (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    cv.putText(right, "Rectified", (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    return np.hstack([left, right])


def process_image(input_path, output_path, K, dist_coeff, show_window=True):
    img = cv.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {input_path}")

    map_cache = {}
    rectified = undistort_frame(img, K, dist_coeff, map_cache)
    compare = stack_compare_view(img, rectified)
    cv.imwrite(output_path, compare)
    print(f"결과 이미지 저장: {output_path}")

    if show_window:
        cv.imshow("Lens Distortion Correction (Image)", compare)
        cv.waitKey(0)
        cv.destroyAllWindows()


def process_images(input_path_or_pattern, output_path, K, dist_coeff, show_window=True):
    if os.path.isdir(input_path_or_pattern):
        pattern = os.path.join(input_path_or_pattern, "*.jpg")
    else:
        pattern = input_path_or_pattern

    input_files = sorted(glob.glob(pattern))
    if not input_files:
        raise FileNotFoundError(f"입력 이미지가 없습니다: {input_path_or_pattern}")

    output_ext = os.path.splitext(output_path)[1]
    output_is_dir = (
        os.path.isdir(output_path)
        or output_path.endswith(("/", "\\"))
        or output_ext == ""
    )
    if output_is_dir:
        os.makedirs(output_path, exist_ok=True)

    map_cache = {}
    for idx, input_file in enumerate(input_files):
        img = cv.imread(input_file)
        if img is None:
            print(f"경고: 이미지를 읽을 수 없어 건너뜁니다: {input_file}")
            continue

        rectified = undistort_frame(img, K, dist_coeff, map_cache)
        compare = stack_compare_view(img, rectified)

        if output_is_dir:
            stem, ext = os.path.splitext(os.path.basename(input_file))
            output_file = os.path.join(output_path, f"{stem}_distortion_demo{ext}")
        elif len(input_files) == 1:
            output_file = output_path
        else:
            out_dir = os.path.dirname(output_path) or "."
            out_name = os.path.basename(output_path)
            stem, ext = os.path.splitext(out_name)
            ext = ext or ".jpg"
            output_file = os.path.join(out_dir, f"{stem}_{idx + 1:02d}{ext}")

        cv.imwrite(output_file, compare)
        print(f"결과 이미지 저장: {output_file}")

        if show_window:
            cv.imshow("Lens Distortion Correction (Image)", compare)
            key = cv.waitKey(400)
            if key == 27:
                print("ESC 입력으로 미리보기를 종료합니다.")
                break

    cv.destroyAllWindows()


def process_video(input_path, output_path, K, dist_coeff, show_window=True):
    video = cv.VideoCapture(input_path)
    if not video.isOpened():
        raise FileNotFoundError(f"동영상을 열 수 없습니다: {input_path}")

    fps = video.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    map_cache = {}
    frame_count = 0

    while True:
        valid, frame = video.read()
        if not valid:
            break

        rectified = undistort_frame(frame, K, dist_coeff, map_cache)
        compare = stack_compare_view(frame, rectified)
        writer.write(compare)
        frame_count += 1

        if show_window:
            cv.imshow("Lens Distortion Correction (Video)", compare)
            key = cv.waitKey(1)
            if key == 27:
                print("ESC 입력으로 재생을 종료합니다.")
                break

    video.release()
    writer.release()
    cv.destroyAllWindows()
    print(f"결과 동영상 저장: {output_path} (총 {frame_count} 프레임)")


def parse_args():
    parser = argparse.ArgumentParser(description="Lens distortion correction using saved calibration")
    parser.add_argument("--mode", choices=["image", "video"], default="image", help="입력 타입")
    parser.add_argument(
        "--input",
        default="data/chessboard/chess_*.jpg",
        help="입력 이미지/패턴/폴더 또는 동영상 경로",
    )
    parser.add_argument(
        "--output",
        default="distortion_results",
        help="출력 경로(이미지 배치 시 폴더 권장)",
    )
    parser.add_argument("--calib", default="data/calibration/calibration_result.npz", help="캘리브레이션 결과 파일")
    parser.add_argument("--no_show", action="store_true", help="미리보기 창 비활성화")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    K, dist_coeff = load_calibration(args.calib)

    if args.mode == "image":
        process_images(args.input, args.output, K, dist_coeff, show_window=not args.no_show)
    else:
        process_video(args.input, args.output, K, dist_coeff, show_window=not args.no_show)