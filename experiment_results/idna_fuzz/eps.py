import pandas as pd
import matplotlib.pyplot as plt

def average_execs_per_sec(filepath, duration_seconds=86400):
    df = pd.read_csv(filepath, header=None, names=["timestamp", "execs", "edges", "execs_per_s", "mem"])
    total_execs = df["execs"].max()
    return total_execs / duration_seconds

def main():
    log_files = ["pythonfuzz.txt", "dqn.txt", "ppo.txt", "grpo.txt"]
    labels = ["Pythonfuzz", "DQN_fuzz", "Rainfuzz_mod", "GRFuzz"]
    avg_execs_list = []

    print("Average execs/s over 24 hours (86400 seconds):\n")

    for file, label in zip(log_files, labels):
        try:
            avg_execs = average_execs_per_sec(file)
            avg_execs_list.append(avg_execs)
            print(f"{label:15}: {avg_execs:.2f} execs/s")
        except Exception as e:
            avg_execs_list.append(0)
            print(f"{label:15}: Error reading {file} - {e}")

    # Vẽ biểu đồ cột
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, avg_execs_list, color='skyblue', edgecolor='black')

    # Thêm giá trị trên đầu cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 10, f'{height:.1f}', ha='center', va='bottom')

    plt.title("Executions per Second of Fuzzing Methods (execs/s)")
    plt.ylabel("Executions per second")
    plt.xlabel("Fuzzing Methods")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("idna_fuzz_eps.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    main()
