#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#  Copyright (C) 2024. Suto-Commune 
#  _
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#  _
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#  _
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
@File       : wechat.py

@Author     : hsn

@Date       : 2024/10/14 下午12:42
"""
import datetime
import decimal
import io
from os import PathLike

import numpy as np
from PIL import ImageEnhance


from ..data import Transaction
from ..scanner import scan
from ...ocr import ocr_normal

def split_lines(image_np, threshold=250):
    # 沿水平方向求取每一列的像素平均值
    column_mean = np.mean(image_np, axis=0)
    # 设定一个阈值来识别空白区域
    split_lines = np.where(column_mean > threshold)[0]

    start = 0
    end = 0
    max_gap = (0, 0)
    for i in split_lines:
        if i - end > 5:
            if end - start > max_gap[1] - max_gap[0]:
                max_gap = (start, end)
            start = i
        end = i
    if end - start > max_gap[1] - max_gap[0]:
        max_gap = (start, end)
    return max_gap[0], max_gap[1]


def median_split(image_np, threshold=250):
    start, end = split_lines(image_np, threshold)
    return (start + end) // 2


def get_transactions(file: io.IOBase | PathLike | str):
    items = []
    for i in scan(file):
        items.append(i)
    
    for img in items:
        img = img.convert("RGB")
        mean_color = np.median(np.array(img.convert("L")), axis=(0, 1))
        if mean_color < 250:
            continue

        icon_rate = 0.8
        img_icon = img.crop((img.height * (1 - icon_rate), img.height * (1 - icon_rate), img.height * icon_rate,
                             img.height * icon_rate))
        img_desc_money = img.crop((img.height * icon_rate, 0, img.width, img.height * 0.5))
        image_np = np.array(img_desc_money.convert("L"))

        sl = median_split(image_np) + img.height * icon_rate

        img_desc = img.crop((img.height * icon_rate, 0, sl, img.height * 0.5))
        img_date = img.crop((img.height * icon_rate, img.height * 0.5, img.width * 0.6, img.height * 0.8))
        img_money = img.crop((sl, 0, img.width, img.height * 0.5))
        date_split, _ = split_lines(np.array(img_date.convert("L")), 254)
        enhancer = ImageEnhance.Contrast(img_date)
        img_date = enhancer.enhance(2.0)
        
        desc: str = ocr_normal.ocr_for_single_line(np.array(img_desc)).get("text").strip()
        date_str: str = ocr_normal.ocr_for_single_line(
            np.array(img_date.crop((0, 0, date_split, img_date.height)))
        ).get("text").strip()
        money_str: str = ocr_normal.ocr_for_single_line(np.array(img_money)).get("text").strip().replace(",", ".")

        date = datetime.datetime.strptime(date_str.replace(" ", ""), "%m月%d日%H:%M")

        if money_str.startswith("-"):
            money_int = -decimal.Decimal(money_str[1:])
        elif money_str.startswith("+"):
            money_int = decimal.Decimal(money_str[1:])
        else:
            print("err", money_str)
            img_money.show()
            money_int = decimal.Decimal(money_str)
        yield Transaction(desc=desc, date=int(date.timestamp() * 1000), money=money_int)
