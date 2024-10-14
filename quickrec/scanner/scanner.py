import io
from os import PathLike
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel


class Item(BaseModel):
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start


def scan(file: io.IOBase | PathLike | str):
    try:
        if not isinstance(file, io.IOBase):
            file = Path(file).open("rb")
        image = Image.open(file)

        image_np = np.array(image.convert("L"))

        sobelx = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
        _, binary = cv2.threshold(np.abs(sobelx), 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        # 6. 查找轮廓，进一步识别线段
        contours, _ = cv2.findContours(np.uint8(dilated), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_lines = []
        min_line_width = 100  # 设置最小线段宽度
        max_aspect_ratio = 20  # 设置最大宽高比，用于过滤非水平线

        # 7. 遍历所有轮廓，计算它们的宽高比，并过滤掉可能的文本
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # 得到边界框

            aspect_ratio = w / h  # 计算宽高比

            # 如果宽度超过设定的最小值且宽高比足够大（即认为是水平线），则保留
            if w > min_line_width and aspect_ratio > max_aspect_ratio:
                valid_lines.append((x, y, w, h))
        lines = []
        # 8. 可视化检测到的线条
        for x, y, w, h in valid_lines:
            lines.append(y + h // 2)
            # cv2.rectangle(image_np, (x, y), (x + w, y + h), 0, 2)  # 用白色矩形标记线条
        gaps = []
        items = []
        for i in range(1, len(lines)):
            gaps.append(lines[i] - lines[i - 1])
            items.append(Item(start=lines[i], end=lines[i - 1]))
        avg_gap = abs(sum(gaps)) / len(gaps)
        cleared_items = []
        for i in items:
            if abs(i.length - avg_gap) < avg_gap / 2:
                cleared_items.append(i)
        # result_image = Image.fromarray(image_np)
        for i in cleared_items:
            # cv2.rectangle(image_np, (0, i.start), (image_np.shape[1], i.end), 0, 2)
            yield image.crop((0, i.start, image.width, i.end))

        # result_image.show()
    finally:
        if not file.closed:
            file.close()
