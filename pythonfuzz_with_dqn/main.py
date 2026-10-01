import argparse
import fuzzer
import importlib
import sys
# Thêm thư mục 'targets' vào sys.path để có thể import module từ đó
sys.path.insert(0, "./targets")

def get_fuzz_function(module_name):
    try:
        module = importlib.import_module(f"targets.{module_name}")
        return module.fuzz
    except (ModuleNotFoundError, AttributeError):
        return None

class PythonFuzz(object):
    def __init__(self, func):
        self.function = func

    def __call__(self, *args, **_):
        parser = argparse.ArgumentParser(description='Coverage-guided fuzzer for python packages')
        parser.add_argument('fuzz_func', type=str, help="Tên của module fuzz cần chạy")
        parser.add_argument('dirs', type=str, nargs='*', help="Thư mục hoặc file để sử dụng làm seed corpus")
        parser.add_argument('--exact-artifact-path', type=str, help='Đặt đường dẫn artifact cụ thể')
        parser.add_argument('--regression', type=bool, default=False, help='Chạy fuzzer để kiểm tra hồi quy')
        parser.add_argument('--rss-limit-mb', type=int, default=4096, help='Giới hạn bộ nhớ (MB)')
        parser.add_argument('--max-input-size', type=int, default=1024, help='Kích thước tối đa của input (bytes)')
        parser.add_argument('--dict', type=str, help='File dictionary')
        parser.add_argument('--close-fd-mask', type=int, default=0, help='Đóng stream output khi khởi động')
        parser.add_argument('--runs', type=int, default=-1, help='Số lần chạy (-1 để chạy vô hạn)')
        parser.add_argument('--timeout', type=int, default=30, help='Timeout mỗi input (giây)')

        args = parser.parse_args()
        fuzz_func = get_fuzz_function(args.fuzz_func)

        if fuzz_func is None:
            print(f"Lỗi: Module '{args.fuzz_func}' không hợp lệ hoặc không tồn tại trong thư mục 'targets'!")
            return

        f = fuzzer.Fuzzer(
            fuzz_func, args.dirs, args.exact_artifact_path,
            args.rss_limit_mb, args.timeout, args.regression, args.max_input_size,
            args.close_fd_mask, args.runs, args.dict
        )
        f.start()

if __name__ == '__main__':
    fuzz_instance = PythonFuzz(None)
    fuzz_instance()
