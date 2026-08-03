import os
import pathlib
import sys

# 修正后的魔数签名（JPEG 只需前两字节）
SIGNATURES = {
    b'\xff\xd8': '.jpg',                # JPEG (兼容 JFIF/EXIF)
    b'\x89PNG\r\n\x1a\n': '.png',       # PNG
    b'GIF87a': '.gif',                  # GIF87a
    b'GIF89a': '.gif',                  # GIF89a
    b'BM': '.bmp',                      # BMP
    b'RIFF': '.webp',                   # 需进一步校验 WEBP
    b'\x00\x00\x01\x00': '.ico',
    b'\x00\x00\x02\x00': '.ico',
    b'II*\x00': '.tiff',               # TIFF
    b'MM\x00*': '.tiff',
}

def detect_extension(file_path):
    """读取文件头，返回真实格式对应的扩展名（如 '.jpg'），无法识别返回 None"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
    except (IOError, PermissionError):
        return None

    for magic, ext in SIGNATURES.items():
        if header.startswith(magic):
            if ext == '.webp':
                # 必须确认是 WEBP (RIFF....WEBP)
                if len(header) >= 12 and header[8:12] == b'WEBP':
                    return '.webp'
                else:
                    continue
            return ext
    return None


def build_target_name(file_path, real_ext):
    """将多后缀文件名重写为仅保留一个正确后缀。"""
    suffixes = file_path.suffixes
    if not suffixes:
        return file_path.with_suffix(real_ext).name

    # 去掉所有旧后缀，只保留基础文件名，再加上真实后缀
    parts = file_path.name.split('.')
    if len(parts) <= len(suffixes):
        base_name = parts[0]
    else:
        base_name = '.'.join(parts[:-len(suffixes)])

    return f"{base_name}{real_ext}"


def normalize_extension(folder_path, recursive=True):
    folder = pathlib.Path(folder_path)
    if not folder.is_dir():
        sys.exit(1)

    all_files = folder.rglob('*') if recursive else folder.glob('*')

    processed = 0
    skipped = 0
    errors = 0

    for file_path in all_files:
        if not file_path.is_file():
            continue

        current_suffix = file_path.suffix.lower()
        real_ext = detect_extension(file_path)
        current_suffixes = [s.lower() for s in file_path.suffixes]


        if real_ext is None:
            skipped += 1
            continue

        if not current_suffixes:
            # 例如文件名是 hello，真实类型是 .png，则改成 hello.png
            new_name = build_target_name(file_path, real_ext)
            new_path = file_path.with_name(new_name)
            action = "添加后缀"
        elif len(current_suffixes) > 1:
            # 例如 foo.jpg.png，改成 foo.png
            new_name = build_target_name(file_path, real_ext)
            new_path = file_path.with_name(new_name)
            action = "修正多后缀"
        elif current_suffix == real_ext.lower():
            skipped += 1
            continue
        else:
            # 例如 foo.jpg，但真实类型是 .png，改成 foo.png
            new_name = build_target_name(file_path, real_ext)
            new_path = file_path.with_name(new_name)
            action = "更正后缀"

        counter = 1
        original_new_path = new_path
        while new_path.exists():
            new_stem = f"{original_new_path.stem}_{counter}"
            new_path = original_new_path.with_name(new_stem + original_new_path.suffix)
            counter += 1

        try:
            file_path.rename(new_path)
            print(f"  -> {action}: {file_path.name} -> {new_path.name}")
            processed += 1
        except Exception as e:
            print(f"  -> 错误: 无法重命名 {file_path.name}: {e}")
            errors += 1

    print(f"\n完成: 已处理 {processed} 个, 跳过 {skipped} 个, 错误 {errors} 个。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
    else:
        target_folder = input("请输入要处理的文件夹路径: ").strip()
    normalize_extension(target_folder, recursive=True)