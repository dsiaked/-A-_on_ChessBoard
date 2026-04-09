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


def build_ar_object_geometry(board_pattern, cell_size):
    # 체스보드 위에 세워진 두께 있는 3D 알파벳 A (빔 3개 조합)
    board_w = (board_pattern[0] - 1) * cell_size
    board_h = (board_pattern[1] - 1) * cell_size

    x_center = board_w * 0.50
    x_left = board_w * 0.33
    x_right = board_w * 0.67

    # A는 x-z 평면에 세운 뒤 y축 방향으로 두께를 주어 입체화
    z_base = -min(board_w, board_h) * 0.04
    z_top = -min(board_w, board_h) * 0.62
    z_cross = -min(board_w, board_h) * 0.36

    cross_left = board_w * 0.43
    cross_right = board_w * 0.57

    y_center = board_h * 0.52
    beam_depth = board_h * 0.14
    beam_width = min(board_w, board_h) * 0.11

    left_base = np.array([x_left, z_base], dtype=np.float32)
    apex = np.array([x_center, z_top], dtype=np.float32)
    right_base = np.array([x_right, z_base], dtype=np.float32)
    cross_l = np.array([cross_left, z_cross], dtype=np.float32)
    cross_r = np.array([cross_right, z_cross], dtype=np.float32)

    beams = [
        (left_base, apex),
        (apex, right_base),
        (cross_l, cross_r),
    ]

    vertices = []
    faces = []
    edges = []
    colors = []

    def add_beam(p0_xz, p1_xz, width, depth, base_color):
        direction = p1_xz - p0_xz
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        hw = width * 0.5
        hd = depth * 0.5

        a = p0_xz + normal * hw
        b = p0_xz - normal * hw
        c = p1_xz - normal * hw
        d = p1_xz + normal * hw

        # 앞면(y+) 4점 + 뒷면(y-) 4점
        local = [
            np.array([a[0], y_center + hd, a[1]], dtype=np.float32),
            np.array([b[0], y_center + hd, b[1]], dtype=np.float32),
            np.array([c[0], y_center + hd, c[1]], dtype=np.float32),
            np.array([d[0], y_center + hd, d[1]], dtype=np.float32),
            np.array([a[0], y_center - hd, a[1]], dtype=np.float32),
            np.array([b[0], y_center - hd, b[1]], dtype=np.float32),
            np.array([c[0], y_center - hd, c[1]], dtype=np.float32),
            np.array([d[0], y_center - hd, d[1]], dtype=np.float32),
        ]

        base = len(vertices)
        vertices.extend(local)

        # 육면체 6면(각 면은 4각형)
        quad_faces = [
            [base + 0, base + 1, base + 2, base + 3],  # front
            [base + 4, base + 5, base + 6, base + 7],  # back
            [base + 0, base + 4, base + 7, base + 3],  # side
            [base + 1, base + 5, base + 6, base + 2],  # side
            [base + 0, base + 1, base + 5, base + 4],  # cap
            [base + 3, base + 2, base + 6, base + 7],  # cap
        ]
        faces.extend(quad_faces)

        # 면별로 명암 차이를 줘 입체감 강화
        b, g, r = base_color
        face_colors = [
            (min(255, b + 35), min(255, g + 35), min(255, r + 35)),
            (max(0, b - 30), max(0, g - 30), max(0, r - 30)),
            (max(0, b - 15), max(0, g - 15), max(0, r - 15)),
            (max(0, b - 45), max(0, g - 45), max(0, r - 45)),
            (min(255, b + 10), min(255, g + 10), min(255, r + 10)),
            (max(0, b - 20), max(0, g - 20), max(0, r - 20)),
        ]
        colors.extend(face_colors)

        beam_edges = [
            (base + 0, base + 1), (base + 1, base + 2), (base + 2, base + 3), (base + 3, base + 0),
            (base + 4, base + 5), (base + 5, base + 6), (base + 6, base + 7), (base + 7, base + 4),
            (base + 0, base + 4), (base + 1, base + 5), (base + 2, base + 6), (base + 3, base + 7),
        ]
        edges.extend(beam_edges)

    add_beam(beams[0][0], beams[0][1], beam_width, beam_depth, (65, 90, 250))
    add_beam(beams[1][0], beams[1][1], beam_width, beam_depth, (65, 90, 250))
    add_beam(beams[2][0], beams[2][1], beam_width, beam_depth * 0.9, (85, 135, 255))

    vertices_arr = np.array(vertices, dtype=np.float32)
    shadow_vertices = vertices_arr.copy()
    shadow_vertices[:, 2] = 0.0
    shadow_vertices[:, 1] += beam_depth * 0.85

    anchor_points = np.array(
        [
            [x_left, y_center + beam_depth * 0.52, z_base],
            [x_center, y_center + beam_depth * 0.52, z_top],
            [x_right, y_center + beam_depth * 0.52, z_base],
            [cross_left, y_center + beam_depth * 0.52, z_cross],
            [cross_right, y_center + beam_depth * 0.52, z_cross],
        ],
        dtype=np.float32,
    )

    return {
        "vertices": vertices_arr,
        "shadow_vertices": shadow_vertices,
        "anchor_points": anchor_points,
        "faces": faces,
        "edges": edges,
        "colors": colors,
    }


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


def draw_letter_a_ar(frame, projected_vertices, rvec, tvec, K, dist_coeff, ar_geometry):
    overlay = frame.copy()

    vertices3d = ar_geometry["vertices"]
    faces = ar_geometry["faces"]
    colors = ar_geometry["colors"]
    edges = ar_geometry["edges"]

    rot_mtx, _ = cv.Rodrigues(rvec)
    camera_pts = (rot_mtx @ vertices3d.T + tvec).T

    # 바닥 그림자 투영
    shadow_vertices_2d = project_points(ar_geometry["shadow_vertices"], rvec, tvec, K, dist_coeff)
    shadow_overlay = frame.copy()
    for face in faces:
        poly = shadow_vertices_2d[np.array(face)]
        cv.fillConvexPoly(shadow_overlay, poly, (30, 30, 30), lineType=cv.LINE_AA)
    cv.addWeighted(shadow_overlay, 0.22, frame, 0.78, 0, frame)

    # 깊이순으로 면을 칠해 간단한 가시성 확보
    face_order = []
    for idx, face in enumerate(faces):
        depth = float(np.mean(camera_pts[np.array(face), 2]))
        face_order.append((depth, idx))
    face_order.sort(reverse=True)

    for _, idx in face_order:
        face = faces[idx]
        poly = projected_vertices[np.array(face)]
        cv.fillConvexPoly(overlay, poly, colors[idx], lineType=cv.LINE_AA)

    cv.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    # 글로우 효과로 시인성 강화
    glow = frame.copy()
    for s, e in edges:
        cv.line(
            glow,
            tuple(projected_vertices[s]),
            tuple(projected_vertices[e]),
            (70, 120, 255),
            12,
            lineType=cv.LINE_AA,
        )
    cv.addWeighted(glow, 0.20, frame, 0.80, 0, frame)

    for s, e in edges:
        cv.line(
            frame,
            tuple(projected_vertices[s]),
            tuple(projected_vertices[e]),
            (255, 245, 210),
            5,
            lineType=cv.LINE_AA,
        )

    # 최종 실루엣을 강조해 어떤 시점에서도 A 형태가 명확하게 보이도록 보강
    anchors2d = project_points(ar_geometry["anchor_points"], rvec, tvec, K, dist_coeff)
    a_edges = [(0, 1), (1, 2), (3, 4)]
    for s, e in a_edges:
        cv.line(
            frame,
            tuple(anchors2d[s]),
            tuple(anchors2d[e]),
            (30, 40, 210),
            16,
            lineType=cv.LINE_AA,
        )
        cv.line(
            frame,
            tuple(anchors2d[s]),
            tuple(anchors2d[e]),
            (250, 220, 120),
            7,
            lineType=cv.LINE_AA,
        )


def render_pose_and_ar(frame, K, dist_coeff, board_pattern, cell_size, ar_geometry):
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

    projected_vertices = project_points(ar_geometry["vertices"], rvec, tvec, K, dist_coeff)
    draw_letter_a_ar(frame, projected_vertices, rvec, tvec, K, dist_coeff, ar_geometry)
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
    ar_geometry = build_ar_object_geometry(board_pattern, cell_size)

    success_count = 0
    for p in files:
        frame = cv.imread(p)
        if frame is None:
            print(f"[SKIP] 이미지를 읽지 못했습니다: {p}")
            continue

        rendered, ok = render_pose_and_ar(frame, K, dist_coeff, board_pattern, cell_size, ar_geometry)
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

    ar_geometry = build_ar_object_geometry(board_pattern, cell_size)
    total = 0
    success = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rendered, pose_ok = render_pose_and_ar(frame, K, dist_coeff, board_pattern, cell_size, ar_geometry)
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

    ar_geometry = build_ar_object_geometry(board_pattern, cell_size)

    print("카메라 실행 중... 종료: ESC")
    total = 0
    success = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rendered, pose_ok = render_pose_and_ar(frame, K, dist_coeff, board_pattern, cell_size, ar_geometry)
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
