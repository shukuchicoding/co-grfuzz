import matplotlib.pyplot as plt
import pandas as pd

# Đọc file log
log_file = "coverage_log.txt"

# Khởi tạo danh sách lưu trữ dữ liệu
timestamps = []
arc_coverage = []

with open(log_file, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue  # Bỏ qua dòng không hợp lệ
        timestamps.append(parts[0])  # Lưu thời gian
        arc_coverage.append(int(parts[2]))  # Lưu arc_coverage

# Chuyển timestamp sang dạng datetime
timestamps = pd.to_datetime(timestamps)

# Vẽ biểu đồ chỉ cho Arc Coverage
plt.figure(figsize=(10, 5))
plt.plot(timestamps, arc_coverage, marker='o', linestyle='-', color="blue", label="Branch Coverage")

# Cấu hình biểu đồ
plt.xlabel("Timestamp")
plt.ylabel("Branch Coverage")
plt.title("Branch Coverage Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plt.savefig("coverage.png")