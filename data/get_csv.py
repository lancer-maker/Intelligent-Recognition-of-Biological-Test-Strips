"""
从文件名自动生成标签 CSV（路径直接写在代码中）
"""
import os
import csv
from pathlib import Path


def parse_filename(filename: str):
    """解析文件名，返回 (标签, 项目ID) 或 None"""
    base, ext = os.path.splitext(filename)
    if ext.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
        return None
    parts = base.split('_')

    # 原有格式 P{患者ID}_{项目ID}_{标签}
    if parts[0].startswith('P') and len(parts) == 3:
        label = parts[2]
        if label in ('0', '1'):
            return label, parts[1]
    # 匿名格式 N_{序号}_{标签}
    if parts[0] == 'N' and len(parts) == 3:
        label = parts[2]
        if label in ('0', '1'):
            return label, ''  # 项目ID留空
    return None


def main():
    # ========== 在这里直接修改路径 ==========
    IMAGE_DIR = r"D:\\software\\vscode\\TOXY\\data\\raw\\N-16"      # 图片所在文件夹
    OUTPUT_CSV = r"D:\\software\\vscode\\TOXY\\data\\labels\\N"  # 输出的CSV文件路径
    # ====================================

    rows = []
    for filepath in Path(IMAGE_DIR).glob('*'):
        if not filepath.is_file():
            continue
        result = parse_filename(filepath.name)
        if result is None:
            continue
        label, item_id = result
        rows.append([filepath.name, label, item_id])

    # 确保输出路径是一个文件，而不是目录
    output_path = Path(OUTPUT_CSV)
    if output_path.suffix == "":
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "labels.csv"
        else:
            output_path = output_path.with_suffix('.csv')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'item_id'])
        writer.writerows(rows)

    print(f"生成完成，共 {len(rows)} 条记录 → {output_path}")


if __name__ == '__main__':
    main()