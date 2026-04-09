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

    if "board_pattern" in data:
        board_pattern = tuple(int(v) for v in data["board_pattern"].tolist())
    else:
        board_pattern = (8, 6)

    if "board_cellsize" in data:
        cell_size = float(data["board_cellsize"])
    else:
        cell_size = 0.025

    return K, dist_coeff, board_pattern, cell_size


def build_object_points(board_pattern, cell_size):
    objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    objp *= cell_size
    return objp


def detect_corners(gray, board_pattern):
    candidate_patterns = [board_pattern]
    swapped = (board_pattern[1], board_pattern[0])
    if swapped != board_pattern:
        candidate_patterns.append(swapped)

    classic_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    sb_flags = cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY

    for pattern in candidate_patterns:
        found, corners = cv.findChessboardCorners(gray, pattern, flags=classic_flags)
        if found:
            return True, corners, pattern

        if hasattr(cv, "findChessboardCornersSB"):
            found_sb, corners_sb = cv.findChessboardCornersSB(gray, pattern, flags=sb_flags)
            if found_sb:
                return True, corners_sb, pattern

    return False, None, None


def build_ar_object_vertices(board_pattern, cell_size):
    # 체스보드 위에 놓인 3D "집"(바닥 사각형 + 지붕) 형태
    board_w = (board_pattern[0] - 1) * cell_size
    board_h = (board_pattern[1] - 1) * cell_size

    margin_x = board_w * 0.15
    margin_y = board_h * 0.15

    x0 = margin_x
    y0 = margin_y
    x1 = board_w - margin_x
    y1 = board_h - margin_y

    wall_h = min(board_w, board_h) * 0.35
    roof_h = wall_h * 0.7

    # z는 카메라 좌표계 상 체스보드 평면에서 바깥쪽으로 튀어나오게 음수 사용
    v0 = np.array([x0, y0, 0], dtype=np.float32)
    v1 = np.array([x1, y0, 0], dtype=np.float32)
    v2 = np.array([x1, y1, 0], dtype=np.float32)
    v3 = np.array([x0, y1, 0], dtype=np.float32)

    v4 = np.array([x0, y0, -wall_h], dtype=np.float32)
    v5 = np.array([x1, y0, -wall_h], dtype=np.float32)
    v6 = np.array([x1, y1, -wall_h], dtype=np.float32)
    v7 = np.array([x0, y1, -wall_h], dtype=np.float32)

    roof_center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, -(wall_h + roof_h)], dtype=np.float32)

    vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7, roof_center], dtype=np.float32)
    return vertices


def project_points(points3d, rvec, tvec, K, dist_coeff):
    points2d, _ = cv.projectPoints(points3d, rvec, tvec, K, dist_coeff)
    return np.int32(points2d.reshape(-1, 2))


def draw_axes(frame, rvec, tvec, K, dist_coeff, cell_size):
    axis_len = 3.0 * cell_size
    axis = np.float32(
        [
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, -axis_len],
        ]
    )
    pts = project_points(axis, rvec, tvec, K, dist_coeff)
    origin = tuple(pts[0])
    cv.line(frame, origin, tuple(pts[1]), (0, 0, 255), 2)  # X-red
    cv.line(frame, origin, tuple(pts[2]), (0, 255, 0), 2)  # Y-green
    cv.line(frame, origin, tuple(pts[3]), (255, 0, 0), 2)  # Z-blue


def draw_house_ar(frame, imgpts):
    v = imgpts

    # 반투명 면 렌더링
    overlay = frame.copy()

    walls = [
        [v[0], v[1], v[5], v[4]],
        [v[1], v[2], v[6], v[5]],
        [v[2], v[3], v[7], v[6]],
        [v[3], v[0], v[4], v[7]],
    ]
    roof = [
        [v[4], v[5], v[8]],
        [v[5], v[6], v[8]],
        [v[6], v[7], v[8]],
        [v[7], v[4], v[8]],
    ]

    for poly in walls:
        cv.fillConvexPoly(overlay, np.array(poly, dtype=np.int32), (80, 180, 255))
    for tri in roof:
        cv.fillConvexPoly(overlay, np.array(tri, dtype=np.int32), (60, 90, 230))

    cv.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # 외곽선
    base_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    side_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    roof_edges = [(4, 8), (5, 8), (6, 8), (7, 8)]

    for s, e in base_edges + top_edges + side_edges + roof_edges:
        cv.line(frame, tuple(v[s]), tuple(v[e]), (20, 40, 120), 2)


def render_pose_and_ar(frame, K, dist_coeff, board_pattern, cell_size, vertices3d):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners, used_pattern = detect_corners(gray, board_pattern)

    if not found:
        cv.putText(
            frame,
            "Chessboard not found",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return frame, False

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = build_object_points(used_pattern, cell_size)
    success, rvec, tvec = cv.solvePnP(objp, corners_refined, K, dist_coeff)
    if not success:
        cv.putText(frame, "solvePnP failed", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, False

    cv.drawChessboardCorners(frame, used_pattern, corners_refined, True)

    imgpts = project_points(vertices3d, rvec, tvec, K, dist_coeff)
    draw_house_ar(frame, imgpts)
    draw_axes(frame, rvec, tvec, K, dist_coeff, cell_size)

    distance_m = float(np.linalg.norm(tvec))
    cv.putText(
        frame,
        f"Pose OK | distance={distance_m:.3f}m",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return frame, True


def process_images(input_pattern, output_dir, K, dist_coeff, board_pattern, cell_size, show=True):
    files = sorted(glob.glob(input_pattern))
    if not files:
        raise FileNotFoundError(f"입력 이미지가 없습니다: {input_pattern}")

    os.makedirs(output_dir, exist_ok=True)
    vertices3d = build_ar_object_vertices(board_pattern, cell_size)

    success_count = 0
    for p in files:
        frame = cv.imread(p)
        if frame is None:
            print(f"[SKIP] 이미지를 읽지 못했습니다: {p}")
            continue

        rendered, ok = render_pose_and_ar(frame, K, dist_coeff, board_pattern, cell_size, vertices3d)
        if ok:
            success_count += 1

        name, ext = os.path.splitext(os.path.basename(p))
        out = os.path.join(output_dir, f"{name}_pose_ar{ext or '.jpg'}")
        cv.imwrite(out, rendered)
        print(f"[SAVE] {out}")

        if show:
            cv.imshow("Pose Estimation + AR (Image)", rendered)
            key = cv.waitKey(400)
            if key == 27:
                print("ESC 입력으로 미리보기를 종료합니다.")
                show = False

    cv.destroyAllWindows()
    print(f"[DONE] 성공 프레임 수: {success_count}/{len(files)}")


def process_video(input_path, output_path, K, dist_coeff, board_pattern, cell_size, show=True):
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"동영상을 열 수 없습니다: {input_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    vertices3d = build_ar_object_vertices(board_pattern, cell_size)
    total = 0
    success = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rendered, pose_ok = render_pose_and_ar(frame, K, dist_coeff, board_pattern, cell_size, vertices3d)
        writer.write(rendered)

        total += 1
        if pose_ok:
            success += 1

        if show:
            cv.imshow("Pose Estimation + AR (Video)", rendered)
            key = cv.waitKey(1)
            if key == 27:
                print("ESC 입력으로 재생을 종료합니다.")
                break

    cap.release()
    writer.release()
    cv.destroyAllWindows()
    print(f"[DONE] 결과 동영상 저장: {output_path} | pose 성공 프레임: {success}/{total}")


def process_webcam(camera_id, output_path, K, dist_coeff, board_pattern, cell_size):
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다. camera_id={camera_id}")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    vertices3d = build_ar_object_vertices(board_pattern, cell_size)

    print("카메라 실행 중... 종료: ESC")
    total = 0
    success = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rendered, pose_ok = render_pose_and_ar(frame, K, dist_coeff, board_pattern, cell_size, vertices3d)
        writer.write(rendered)

        total += 1
        if pose_ok:
            success += 1

        cv.imshow("Pose Estimation + AR (Webcam)", rendered)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()
    print(f"[DONE] 결과 동영상 저장: {output_path} | pose 성공 프레임: {success}/{total}")


def parse_args():
    parser = argparse.ArgumentParser(description="Camera pose estimation and AR object visualization")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], default="image", help="실행 모드")
    parser.add_argument("--input", default="data/chessboard/chess_*.jpg", help="입력 이미지 패턴 또는 동영상 경로")
    parser.add_argument("--output", default="ar_results", help="출력 폴더(이미지) 또는 파일(동영상)")
    parser.add_argument("--calib", default="data/calibration/calibration_result.npz", help="캘리브레이션 파일 경로")
    parser.add_argument("--board_cols", type=int, default=None, help="내부 코너 가로 개수(옵션)")
    parser.add_argument("--board_rows", type=int, default=None, help="내부 코너 세로 개수(옵션)")
    parser.add_argument("--cell_size", type=float, default=None, help="체스보드 한 칸 크기(옵션)")
    parser.add_argument("--camera_id", type=int, default=0, help="웹캠 ID")
    parser.add_argument("--no_show", action="store_true", help="미리보기 창 비활성화")
    return parser.parse_args()


def main():
    args = parse_args()
    K, dist_coeff, board_pattern, cell_size = load_calibration(args.calib)

    if args.board_cols is not None and args.board_rows is not None:
        board_pattern = (args.board_cols, args.board_rows)
    if args.cell_size is not None:
        cell_size = args.cell_size

    if args.mode == "image":
        process_images(
            args.input,
            args.output,
            K,
            dist_coeff,
            board_pattern,
            cell_size,
            show=not args.no_show,
        )
        return

    if args.mode == "video":
        output_video = args.output
        output_ext = os.path.splitext(output_video)[1].lower()
        if output_ext not in (".mp4", ".avi", ".mov", ".mkv"):
            output_video = os.path.join(output_video, "pose_ar_video.mp4")
        process_video(
            args.input,
            output_video,
            K,
            dist_coeff,
            board_pattern,
            cell_size,
            show=not args.no_show,
        )
        return

    output_video = args.output
    output_ext = os.path.splitext(output_video)[1].lower()
    if output_ext not in (".mp4", ".avi", ".mov", ".mkv"):
        output_video = os.path.join(output_video, "pose_ar_webcam.mp4")

    process_webcam(
        args.camera_id,
        output_video,
        K,
        dist_coeff,
        board_pattern,
        cell_size,
    )


if __name__ == "__main__":
    main()
