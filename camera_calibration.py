import argparse
import glob
import os

import cv2 as cv
import numpy as np


def collect_images(image_dir, patterns=("*.jpg", "*.jpeg", "*.png", "*.bmp")):
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    image_paths = sorted(image_paths)
    return image_paths


def build_object_points(board_pattern, board_cellsize):
    objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    objp *= board_cellsize
    return objp


def detect_chessboard_corners(gray, board_pattern):
    candidate_patterns = [board_pattern]
    swapped_pattern = (board_pattern[1], board_pattern[0])
    if swapped_pattern != board_pattern:
        candidate_patterns.append(swapped_pattern)

    classic_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    sb_flags = cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY

    for pattern in candidate_patterns:
        found, corners = cv.findChessboardCorners(gray, pattern, flags=classic_flags)
        if found:
            return True, corners, pattern, "classic"

        if hasattr(cv, "findChessboardCornersSB"):
            found_sb, corners_sb = cv.findChessboardCornersSB(gray, pattern, flags=sb_flags)
            if found_sb:
                return True, corners_sb, pattern, "SB"

    return False, None, None, None


def calib_camera_from_chessboard(image_paths, board_pattern, board_cellsize, show_preview=False):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp_cache = {
        board_pattern: build_object_points(board_pattern, board_cellsize)
    }
    swapped_pattern = (board_pattern[1], board_pattern[0])
    if swapped_pattern != board_pattern:
        objp_cache[swapped_pattern] = build_object_points(swapped_pattern, board_cellsize)

    obj_points = []
    img_points = []
    used_files = []
    image_size = None

    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            print(f"[WARN] 이미지를 읽지 못했습니다: {path}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners, used_pattern, method = detect_chessboard_corners(gray, board_pattern)

        if not found:
            print(f"[SKIP] 코너 검출 실패: {path}")
            continue

        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp_cache[used_pattern].copy())
        img_points.append(corners_refined)
        used_files.append(path)
        image_size = gray.shape[::-1]
        print(f"[OK] 코너 검출 성공({method}, pattern={used_pattern[0]}x{used_pattern[1]}): {path}")

        if show_preview:
            preview = img.copy()
            cv.drawChessboardCorners(preview, used_pattern, corners_refined, found)
            cv.imshow("Detected Corners", preview)
            key = cv.waitKey(300)
            if key == 27:
                show_preview = False

    if show_preview:
        cv.destroyAllWindows()

    if len(img_points) < 3:
        raise RuntimeError(
            "캘리브레이션에 충분한 유효 이미지가 없습니다. 최소 3장 이상 필요합니다.\n"
            f"현재 패턴: {board_pattern}. 내부 코너 개수(가로 x 세로)를 다시 확인하세요.\n"
            "예시: --board_cols 5 --board_rows 8 또는 --board_cols 8 --board_rows 5"
        )

    rms, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )

    # 과제 보고용 재투영 오차(mean reprojection error) 계산
    total_error = 0.0
    for i in range(len(obj_points)):
        projected, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist_coeff)
        error = cv.norm(img_points[i], projected, cv.NORM_L2) / len(projected)
        total_error += error
    mean_reproj_error = total_error / len(obj_points)

    return rms, mean_reproj_error, K, dist_coeff, used_files


def save_calibration(output_path, board_pattern, board_cellsize, rms, mean_reproj_error, K, dist_coeff, used_files):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        output_path,
        board_pattern=np.array(board_pattern, dtype=np.int32),
        board_cellsize=np.float32(board_cellsize),
        rms=np.float64(rms),
        mean_reprojection_error=np.float64(mean_reproj_error),
        K=K,
        dist_coeff=dist_coeff,
        used_files=np.array(used_files),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Chessboard images based camera calibration")
    parser.add_argument("--image_dir", default="data/chessboard", help="체스보드 이미지 폴더")
    parser.add_argument("--board_cols", type=int, default=8, help="내부 코너 가로 개수 (예: 사각형 9칸이면 8)")
    parser.add_argument("--board_rows", type=int, default=6, help="내부 코너 세로 개수 (예: 사각형 7칸이면 6)")
    parser.add_argument("--cell_size", type=float, default=0.025, help="체스보드 한 칸 실제 크기(미터)")
    parser.add_argument("--output", default="data/calibration/calibration_result.npz", help="캘리브레이션 결과 저장 파일")
    parser.add_argument("--preview", action="store_true", help="코너 검출 미리보기 표시")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    board_pattern = (args.board_cols, args.board_rows)

    image_paths = collect_images(args.image_dir)
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"이미지 폴더에 파일이 없습니다: {args.image_dir}\n"
            "예시: data/chessboard/chess_01.jpg, chess_02.jpg ..."
        )

    print(f"입력 이미지 수: {len(image_paths)}")
    rms, mean_reproj_error, K, dist_coeff, used_files = calib_camera_from_chessboard(
        image_paths, board_pattern, args.cell_size, show_preview=args.preview
    )

    save_calibration(
        args.output,
        board_pattern,
        args.cell_size,
        rms,
        mean_reproj_error,
        K,
        dist_coeff,
        used_files,
    )

    print("\n=== Camera Calibration Results ===")
    print(f"* 전체 입력 이미지 수 = {len(image_paths)}")
    print(f"* 실제 사용된 이미지 수 = {len(used_files)}")
    print(f"* RMS error = {rms:.6f}")
    print(f"* Mean reprojection error = {mean_reproj_error:.6f}")
    print(f"* fx = {K[0, 0]:.6f}, fy = {K[1, 1]:.6f}, cx = {K[0, 2]:.6f}, cy = {K[1, 2]:.6f}")
    print("* Camera matrix (K) =")
    print(K)
    print("* Distortion coefficient (k1, k2, p1, p2, k3, ...) =")
    print(dist_coeff.flatten())
    print(f"\n캘리브레이션 결과 저장 완료: {args.output}")