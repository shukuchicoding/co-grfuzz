import pandas as pd
import matplotlib.pyplot as plt

def process_log_continuous(filepath):
    df = pd.read_csv(filepath, header=None, names=["timestamp", "execs", "edges", "execs_per_s", "mem"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_time = df["timestamp"].iloc[0]
    df["elapsed_hours"] = df["timestamp"].apply(lambda t: (t - start_time).total_seconds() / 3600)
    df = df[df["elapsed_hours"] <= 24]
    return df["elapsed_hours"], df["edges"]

# Tên file log và label sẽ điều chỉnh ở đây.

log_files = ["pythonfuzz.txt", "dqn.txt", "ppo.txt", "grpo.txt"]
labels = ["Pythonfuzz", "PyDQNfuzz", "PyRainfuzz", "GRFuzz"]
colors = ["blue", "green", "red", "orange"]

plt.figure(figsize=(10, 6))

min_coverage = float('inf')
max_coverage = 0

for file, label, color in zip(log_files, labels, colors):
    x, y = process_log_continuous(file)
    plt.plot(x, y, label=label, color=color)
    min_coverage = min(min_coverage, y.min())
    max_coverage = max(max_coverage, y.max())

# Giới hạn trục Y theo dữ liệu
plt.ylim(
    bottom=90,  # chừa khoảng dưới 5% hoặc 50 đơn vị
    top=2400
)

plt.xlabel("Elapsed Time (hours)")
plt.ylabel("Edge Coverage")
plt.title("Coverage over Time")
plt.xlim(0, 24)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("pyyaml_fuzz_cov.pdf", format="pdf")
plt.show()
