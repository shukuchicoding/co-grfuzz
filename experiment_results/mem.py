import pandas as pd
import matplotlib.pyplot as plt

def average_memory(filepath):
    df = pd.read_csv(filepath, header=None, names=["timestamp", "execs", "edges", "execs_per_s", "mem"])
    return df["mem"].mean()

def main():
    log_files = ["pythonfuzz.txt", "dqn.txt", "ppo.txt", "grpo.txt"]
    labels = ["Pythonfuzz", "DQN_fuzz", "Rainfuzz_mod", "GRFuzz"]
    avg_memory_list = []

    print("Average memory consumption (MB):\n")

    for file, label in zip(log_files, labels):
        try:
            avg_mem = average_memory(file)
            avg_memory_list.append(avg_mem)
            print(f"{label:15}: {avg_mem:.2f} MB")
        except Exception as e:
            avg_memory_list.append(0)
            print(f"{label:15}: Error reading {file} - {e}")

    # Vẽ biểu đồ cột
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, avg_memory_list, color='lightgreen', edgecolor='black')

    # Thêm giá trị trên đầu cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.1f}', ha='center', va='bottom')

    plt.title("Memory consumption of Fuzzing Methods (MiB)")
    plt.ylabel("Memory consumption")
    plt.xlabel("Fuzzing Methods")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("mem.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    main()
