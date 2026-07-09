"""
通用表格单元格疑似漏填检测 Web 小工具
运行：
    pip install streamlit pymupdf opencv-python numpy pillow
    streamlit run generic_check_streamlit.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math

import cv2
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from PIL import Image

Box = Tuple[int, int, int, int]

@dataclass
class CellCheckResult:
    page: int
    cell_index: int
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    mean_gray: float
    dark_pixels: int
    dark_ratio: float
    component_count: int
    status: str
# =========================
# 固定参数：只保留“多显示疑似”这一套检测逻辑
# =========================
PROFILE = {
    "dpi": 160,
    "line_bin_threshold": 205,
    "horizontal_kernel_ratio": 0.070,
    "vertical_kernel_ratio": 0.070,
    "min_h_coverage": 0.08,
    "min_v_coverage": 0.08,
    "line_merge_gap": 12,
    "min_cell_width": 16,
    "min_cell_height": 12,
    "edge_presence_threshold": 0.26,
    "internal_line_presence": 0.68,
    "max_col_span": 7,
    "max_row_span": 9,
    "content_threshold": 188,
    "inner_margin_ratio": 0.075,
    "dark_ratio_threshold": 0.0045,
    "min_dark_pixels": 24,
    "min_components": 2,
    "skip_gray_cells": True,
    "gray_mean_limit": 225,
    "gray_std_limit": 38,
}

# =========================
# PDF 与基础图像工具
# =========================
def iter_pdf_pages(pdf_bytes: bytes, dpi: int = 150):
    """把 PDF 每页渲染为 OpenCV BGR 图片。"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    total_pages = doc.page_count
    try:
        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield page_index + 1, total_pages, img
    finally:
        doc.close()

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def resize_for_web(img: np.ndarray, max_width: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_size = (max_width, max(1, int(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def threshold_dark(gray: np.ndarray, threshold: int = 200) -> np.ndarray:
    return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]

def adaptive_dark(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )

def cluster_positions(values: np.ndarray | Sequence[int], max_gap: int = 8) -> List[int]:
    if len(values) == 0:
        return []
    arr = np.sort(np.asarray(values, dtype=int))
    out: List[int] = []
    start = prev = int(arr[0])
    for v in arr[1:]:
        v = int(v)
        if v - prev > max_gap:
            out.append((start + prev) // 2)
            start = v
        prev = v
    out.append((start + prev) // 2)
    return out

def rotate_bound_white(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.05:
        return img
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - center[0]
    m[1, 2] += new_h / 2 - center[1]
    return cv2.warpAffine(
        img,
        m,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

# =========================
# 纠偏与主表区域定位
# =========================
def estimate_skew_angle(img: np.ndarray, bin_threshold: int = 205, max_abs_angle: float = 4.0) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = threshold_dark(gray, bin_threshold)
    h, w = binary.shape[:2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, int(w * 0.12)), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    lines = cv2.HoughLinesP(
        horizontal,
        rho=1,
        theta=np.pi / 180,
        threshold=max(60, int(w * 0.04)),
        minLineLength=max(80, int(w * 0.18)),
        maxLineGap=max(8, int(w * 0.015)),
    )
    if lines is None:
        return 0.0

    angles: List[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        length = math.hypot(dx, dy)
        if length < w * 0.15:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) <= max_abs_angle:
            angles.append(angle)

    if not angles:
        return 0.0
    return -float(np.median(angles))


def extract_line_masks(
    img: np.ndarray,
    bin_threshold: int,
    horizontal_kernel_ratio: float,
    vertical_kernel_ratio: float,
    adaptive: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = adaptive_dark(gray) if adaptive else threshold_dark(gray, bin_threshold)
    h, w = binary.shape[:2]
    h_len = max(20, int(w * horizontal_kernel_ratio))
    v_len = max(20, int(h * vertical_kernel_ratio))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
    vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    line_mask = cv2.bitwise_or(horizontal, vertical)
    return horizontal, vertical, line_mask


def auto_crop_main_table(
    img: np.ndarray,
    bin_threshold: int,
    horizontal_kernel_ratio: float,
    vertical_kernel_ratio: float,
    margin_ratio: float = 0.010,
) -> Tuple[np.ndarray, Box]:
    h, w = img.shape[:2]
    _, _, line_mask = extract_line_masks(
        img,
        bin_threshold=bin_threshold,
        horizontal_kernel_ratio=horizontal_kernel_ratio,
        vertical_kernel_ratio=vertical_kernel_ratio,
        adaptive=True,
    )

    safe = np.zeros_like(line_mask)
    x_margin = int(w * 0.01)
    y_top = int(h * 0.035)
    y_bottom = int(h * 0.90)
    safe[y_top:y_bottom, x_margin : w - x_margin] = line_mask[y_top:y_bottom, x_margin : w - x_margin]

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, int(w * 0.012)), max(9, int(h * 0.008))),
    )
    connected = cv2.dilate(safe, connect_kernel, iterations=2)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[int, int, int, int, int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if cw < int(w * 0.30) or ch < int(h * 0.035):
            continue
        if area < int(w * h * 0.012):
            continue
        if ch < int(h * 0.030):
            continue
        candidates.append((x, y, cw, ch, area))

    if candidates:
        x, y, cw, ch, _ = max(candidates, key=lambda t: t[4])
        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)
    else:
        ys, xs = np.where(safe > 0)
        if len(xs) == 0:
            return img, (0, 0, w, h)
        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio)
        x1 = max(0, int(xs.min()) - pad_x)
        y1 = max(0, int(ys.min()) - pad_y)
        x2 = min(w, int(xs.max()) + pad_x)
        y2 = min(h, int(ys.max()) + pad_y)

    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

# =========================
# 横竖线聚类与单元格生成
# =========================
def get_grid_lines_from_masks(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    min_h_coverage: float,
    min_v_coverage: float,
    line_merge_gap: int,
) -> Tuple[List[int], List[int]]:
    h, w = horizontal.shape[:2]
    h_proj = horizontal.sum(axis=1) / 255
    v_proj = vertical.sum(axis=0) / 255

    ys = np.where(h_proj >= max(6, w * min_h_coverage))[0]
    xs = np.where(v_proj >= max(6, h * min_v_coverage))[0]

    y_lines = cluster_positions(ys, max_gap=line_merge_gap)
    x_lines = cluster_positions(xs, max_gap=line_merge_gap)

    y_lines = sorted(set(int(y) for y in y_lines if 0 <= y < h))
    x_lines = sorted(set(int(x) for x in x_lines if 0 <= x < w))
    return x_lines, y_lines


def horizontal_edge_presence(horizontal: np.ndarray, y: int, x1: int, x2: int, tol: int = 4) -> float:
    h, w = horizontal.shape[:2]
    if x2 <= x1:
        return 0.0
    yy1 = max(0, y - tol)
    yy2 = min(h, y + tol + 1)
    xx1 = max(0, x1)
    xx2 = min(w, x2)
    roi = horizontal[yy1:yy2, xx1:xx2]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).any(axis=0).mean())


def vertical_edge_presence(vertical: np.ndarray, x: int, y1: int, y2: int, tol: int = 4) -> float:
    h, w = vertical.shape[:2]
    if y2 <= y1:
        return 0.0
    xx1 = max(0, x - tol)
    xx2 = min(w, x + tol + 1)
    yy1 = max(0, y1)
    yy2 = min(h, y2)
    roi = vertical[yy1:yy2, xx1:xx2]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).any(axis=1).mean())


def has_full_internal_horizontal(
    horizontal: np.ndarray,
    x1: int,
    x2: int,
    y_lines: Sequence[int],
    y_top: int,
    y_bottom: int,
    min_presence: float,
) -> bool:
    for y in y_lines:
        if y_top + 3 < y < y_bottom - 3:
            if horizontal_edge_presence(horizontal, y, x1, x2) >= min_presence:
                return True
    return False


def has_full_internal_vertical(
    vertical: np.ndarray,
    y1: int,
    y2: int,
    x_lines: Sequence[int],
    x_left: int,
    x_right: int,
    min_presence: float,
) -> bool:
    for x in x_lines:
        if x_left + 3 < x < x_right - 3:
            if vertical_edge_presence(vertical, x, y1, y2) >= min_presence:
                return True
    return False


def deduplicate_boxes(boxes: List[Box], iou_threshold: float = 0.85) -> List[Box]:
    def area(b: Box) -> int:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def iou(a: Box, b: Box) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = area((ix1, iy1, ix2, iy2))
        union = area(a) + area(b) - inter
        return inter / union if union > 0 else 0.0

    boxes = sorted(boxes, key=lambda b: (b[1], b[0], area(b)))
    kept: List[Box] = []
    for b in boxes:
        if any(iou(b, k) > iou_threshold for k in kept):
            continue
        kept.append(b)
    return kept


def generate_cells_from_grid(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    x_lines: Sequence[int],
    y_lines: Sequence[int],
    min_cell_width: int,
    min_cell_height: int,
    edge_presence_threshold: float,
    internal_line_presence: float,
    max_col_span: int,
    max_row_span: int,
) -> List[Box]:
    xs = list(x_lines)
    ys = list(y_lines)
    if len(xs) < 2 or len(ys) < 2:
        return []

    boxes: List[Box] = []
    used_top_left: set[Tuple[int, int]] = set()

    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            if (xi, yi) in used_top_left:
                continue

            best: Optional[Box] = None
            max_yj = min(len(ys), yi + max_row_span + 1)
            max_xj = min(len(xs), xi + max_col_span + 1)

            for yj in range(yi + 1, max_yj):
                if best is not None:
                    break
                for xj in range(xi + 1, max_xj):
                    x1, x2 = xs[xi], xs[xj]
                    y1, y2 = ys[yi], ys[yj]
                    if x2 - x1 < min_cell_width or y2 - y1 < min_cell_height:
                        continue

                    top_ok = horizontal_edge_presence(horizontal, y1, x1, x2) >= edge_presence_threshold
                    bottom_ok = horizontal_edge_presence(horizontal, y2, x1, x2) >= edge_presence_threshold
                    left_ok = vertical_edge_presence(vertical, x1, y1, y2) >= edge_presence_threshold
                    right_ok = vertical_edge_presence(vertical, x2, y1, y2) >= edge_presence_threshold
                    if not (top_ok and bottom_ok and left_ok and right_ok):
                        continue

                    if has_full_internal_horizontal(horizontal, x1, x2, ys, y1, y2, internal_line_presence):
                        continue
                    if has_full_internal_vertical(vertical, y1, y2, xs, x1, x2, internal_line_presence):
                        continue

                    best = (x1, y1, x2, y2)
                    break

            if best is not None:
                boxes.append(best)
                used_top_left.add((xi, yi))

    return deduplicate_boxes(boxes)


# =========================
# 单元格内容占比检测
# =========================
def remove_table_lines_for_cell(binary: np.ndarray) -> np.ndarray:
    """在单元格内部去掉残留边框线，保留文字、手写、勾、斜杠等内容。"""
    h, w = binary.shape[:2]
    if h <= 0 or w <= 0:
        return binary
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * 0.60)), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.60))))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(horizontal, vertical)
    ink = cv2.subtract(binary, lines)
    # 只去很小的独立噪点，不要过度开运算，否则复选框里的浅痕迹可能被抹掉。
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return ink


def is_gray_or_filled_background(img: np.ndarray, box: Box, gray_mean_limit: float, gray_std_limit: float) -> bool:
    """检测灰底/底纹格，通常不按空白格处理。"""
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    px = max(2, int(w * 0.08))
    py = max(2, int(h * 0.08))
    roi = img[y1 + py : y2 - py, x1 + px : x2 - px]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    return mean < gray_mean_limit and std < gray_std_limit


def cell_content_metrics(
    img: np.ndarray,
    box: Box,
    content_threshold: int,
    inner_margin_ratio: float,
) -> Tuple[float, int, float, int]:
    x1, y1, x2, y2 = box
    h_img, w_img = img.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    mx = max(1, int(w * inner_margin_ratio))
    my = max(1, int(h * inner_margin_ratio))
    ix1, ix2 = x1 + mx, x2 - mx
    iy1, iy2 = y1 + my, y2 - my
    if ix2 <= ix1 or iy2 <= iy1:
        return 255.0, 0, 0.0, 0

    roi = img[iy1:iy2, ix1:ix2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_gray = float(gray.mean())

    # 同时使用固定阈值和自适应阈值：
    # 固定阈值对印刷/手写较稳；自适应阈值对扫描阴影、浅勾选痕迹更友好。
    binary_fixed = threshold_dark(gray, content_threshold)
    binary_adaptive = adaptive_dark(gray)
    binary = cv2.bitwise_or(binary_fixed, binary_adaptive)

    ink = remove_table_lines_for_cell(binary)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cleaned = np.zeros_like(ink)
    component_count = 0
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 4:
            continue
        if bw <= 2 and bh <= 2:
            continue
        component_count += 1
        cleaned[labels == i] = 255

    dark_pixels = int(cv2.countNonZero(cleaned))
    dark_ratio = dark_pixels / max(1, cleaned.size)
    return mean_gray, dark_pixels, dark_ratio, component_count


def check_cells_on_page(
    img: np.ndarray,
    page_index: int,
    cells: Sequence[Box],
    profile: dict,
) -> List[CellCheckResult]:
    results: List[CellCheckResult] = []
    for idx, box in enumerate(cells, start=1):
        x1, y1, x2, y2 = box
        if profile["skip_gray_cells"] and is_gray_or_filled_background(
            img,
            box,
            profile["gray_mean_limit"],
            profile["gray_std_limit"],
        ):
            continue

        mean_gray, dark_pixels, dark_ratio, component_count = cell_content_metrics(
            img,
            box,
            content_threshold=profile["content_threshold"],
            inner_margin_ratio=profile["inner_margin_ratio"],
        )

        # 疑似检测逻辑：
        # 只要单元格里黑色痕迹达到任一条件，就先认为“有内容”；否则红框提示。
        # “多显示疑似”模式会自动提高这些条件，从而把更多弱痕迹格子列为疑似。
        has_content = (
            dark_ratio >= profile["dark_ratio_threshold"]
            or dark_pixels >= profile["min_dark_pixels"]
            or component_count >= profile["min_components"]
        )
        status = "有内容/达标" if has_content else "疑似空白/漏填"
        results.append(
            CellCheckResult(
                page=page_index,
                cell_index=idx,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                width=x2 - x1,
                height=y2 - y1,
                mean_gray=mean_gray,
                dark_pixels=dark_pixels,
                dark_ratio=dark_ratio,
                component_count=component_count,
                status=status,
            )
        )
    return results


def annotate_page(img: np.ndarray, results: Sequence[CellCheckResult], show_green_cells: bool) -> np.ndarray:
    annotated = img.copy()
    for r in results:
        is_problem = r.status == "疑似空白/漏填"
        if not is_problem and not show_green_cells:
            continue
        color = (0, 0, 255) if is_problem else (0, 170, 0)
        thickness = 3 if is_problem else 1
        cv2.rectangle(annotated, (r.x1, r.y1), (r.x2, r.y2), color, thickness)
        if is_problem:
            label = f"{r.cell_index}"
            cv2.putText(
                annotated,
                label,
                (r.x1 + 2, max(16, r.y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def draw_detected_grid(img: np.ndarray, x_lines: Sequence[int], y_lines: Sequence[int], cells: Sequence[Box]) -> np.ndarray:
    out = img.copy()
    for x in x_lines:
        cv2.line(out, (x, 0), (x, out.shape[0] - 1), (255, 0, 0), 1)
    for y in y_lines:
        cv2.line(out, (0, y), (out.shape[1] - 1, y), (255, 0, 0), 1)
    for b in cells:
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (0, 180, 0), 1)
    return out


# =========================
# Streamlit 页面
# =========================
st.set_page_config(page_title="表格疑似漏填检测", layout="wide")
st.title("检查单元格是否漏写的小小web网站")
st.caption("上传扫描 PDF 后，只显示存在疑似空白/漏填单元格的页面。红框=疑似，绿框=识别到有内容。")

with st.sidebar:
    enable_deskew = st.checkbox(
        "扫描歪斜时启用纠偏",
        value=False,
        help="纠偏会更慢。表格明显倾斜时再打开。",
    )
    show_green_cells = st.checkbox(
        "疑似页面显示绿色已填格",
        value=True,
        help="打开后能看到红绿框，便于判断单元格切分是否合理；关闭后页面更清爽。",
    )
    show_debug_grid = st.checkbox(
        "调试：显示网格线",
        value=False,
        help="识别不准时再打开，用来看蓝色网格线和绿色单元格框是否切对。",
    )

profile = PROFILE.copy()

uploaded = st.file_uploader("上传待检查作业单 PDF", type=["pdf"])
if uploaded is None:
    st.info("请上传扫描版或图片版 PDF。")
    st.stop()

pdf_bytes = uploaded.read()

try:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    doc.close()
except Exception as exc:
    st.error(f"PDF 读取失败：{exc}")
    st.stop()

st.success(f"当前 PDF 共 {total_pages} 页。渲染 DPI：{profile['dpi']}。")

progress = st.progress(0, text="开始检测……")
problem_page_count = 0
checked_page_count = 0

try:
    for page_index, total_pages, img in iter_pdf_pages(pdf_bytes, dpi=profile["dpi"]):
        checked_page_count += 1
        progress.progress(page_index / total_pages, text=f"正在检测第 {page_index}/{total_pages} 页……")

        working = img
        angle = 0.0
        if enable_deskew:
            angle = estimate_skew_angle(working, bin_threshold=profile["line_bin_threshold"])
            working = rotate_bound_white(working, angle)

        table_img, bbox = auto_crop_main_table(
            working,
            bin_threshold=profile["line_bin_threshold"],
            horizontal_kernel_ratio=profile["horizontal_kernel_ratio"],
            vertical_kernel_ratio=profile["vertical_kernel_ratio"],
        )

        horizontal, vertical, _ = extract_line_masks(
            table_img,
            bin_threshold=profile["line_bin_threshold"],
            horizontal_kernel_ratio=profile["horizontal_kernel_ratio"],
            vertical_kernel_ratio=profile["vertical_kernel_ratio"],
            adaptive=True,
        )
        x_lines, y_lines = get_grid_lines_from_masks(
            horizontal,
            vertical,
            min_h_coverage=profile["min_h_coverage"],
            min_v_coverage=profile["min_v_coverage"],
            line_merge_gap=profile["line_merge_gap"],
        )

        cells = generate_cells_from_grid(
            horizontal,
            vertical,
            x_lines=x_lines,
            y_lines=y_lines,
            min_cell_width=profile["min_cell_width"],
            min_cell_height=profile["min_cell_height"],
            edge_presence_threshold=profile["edge_presence_threshold"],
            internal_line_presence=profile["internal_line_presence"],
            max_col_span=profile["max_col_span"],
            max_row_span=profile["max_row_span"],
        )

        if not cells:
            if show_debug_grid:
                st.warning(f"第 {page_index} 页未识别到单元格。")
                st.image(bgr_to_pil(resize_for_web(table_img)), caption=f"第 {page_index} 页裁剪后的主表区域", use_container_width=True)
            continue

        results = check_cells_on_page(table_img, page_index=page_index, cells=cells, profile=profile)
        problem_results = [r for r in results if r.status == "疑似空白/漏填"]

        # 核心要求：没有疑似的页面不显示，减少网页压力。
        if not problem_results:
            continue

        problem_page_count += 1
        st.subheader(f"第 {page_index} 页：发现 {len(problem_results)} 个疑似单元格")
        if show_debug_grid:
            st.caption(f"识别单元格：{len(cells)} 个；纠偏角度：{angle:.2f}°；主表裁剪框：{bbox}")

        annotated = annotate_page(table_img, results, show_green_cells=show_green_cells)
        st.image(bgr_to_pil(resize_for_web(annotated)), use_container_width=True)

        if show_debug_grid:
            debug = draw_detected_grid(table_img, x_lines, y_lines, cells)
            st.image(
                bgr_to_pil(resize_for_web(debug)),
                caption="调试图：蓝线=聚类网格线，绿框=识别到的单元格",
                use_container_width=True,
            )

    progress.empty()
except Exception as exc:
    st.error(f"检测过程中出错：{exc}")
    st.stop()

st.markdown("---")
if problem_page_count == 0:
    st.success("检测结束：当前参数下没有发现疑似空白/漏填页面。")
else:
    st.warning(f"检测结束：共发现 {problem_page_count} 页存在疑似空白/漏填单元格。")

st.caption("说明：本工具只负责把疑似单元格提出来，不判断是否必须填写正确；最终建议人工复核。")
