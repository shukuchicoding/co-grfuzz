import yaml
import traceback

def fuzz(buf):
    # try:
        result = yaml.safe_load(buf)
    # except yaml.YAMLError:
    #     traceback.print_exc()

fuzz("0b_")
# import os

# def read_all_files_in_current_directory():
#     current_dir = os.path.dirname(os.path.abspath(__file__))  # Thư mục chứa file .py
#     for filename in os.listdir(current_dir):
#         filepath = os.path.join(current_dir, filename)

#         # Bỏ qua thư mục và chính file script
#         if os.path.isfile(filepath) and filename != os.path.basename(__file__):
#             try:
#                 with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
#                     content = f.read()
#                     # print(f"--- Nội dung của file: {filename} ---")
#                     print(content[:1000])  # In ra 1000 ký tự đầu tiên (tránh quá dài)
#                     print("\n")
#             except Exception as e:
#                 print(f"[!] Lỗi đọc file {filename}: {e}")

# read_all_files_in_current_directory()