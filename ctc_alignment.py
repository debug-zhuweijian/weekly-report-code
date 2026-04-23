"""
ctc_alignment.py -- CTC 路径枚举 + collapse + 概率求和验证
对应周报章节：时序对齐问题的进一步理解
"""

import itertools


def collapse(path):
    merged = []
    prev = None
    for p in path:
        if p != prev:
            merged.append(p)
        prev = p
    return "".join(p for p in merged if p != "-")


def main():
    labels = ["-", "A", "B"]
    probs = [
        {"-": 0.4, "A": 0.5, "B": 0.1},
        {"-": 0.4, "A": 0.5, "B": 0.1},
        {"-": 0.7, "A": 0.2, "B": 0.1},
    ]
    T = len(probs)

    print(f"CTC alignment enumeration")
    print(f"Vocabulary: {labels}")
    print(f"Timesteps: {T}")
    print(f"Target: 'A'\n")

    all_paths = list(itertools.product(labels, repeat=T))
    print(f"Total possible paths: {len(all_paths)}\n")

    print(f"All paths with collapse result and probability:")
    print(f"  {'Path':>15}  {'Collapsed':>10}  {'Prob':>10}")
    path_data = []
    for path in all_paths:
        p = 1.0
        for t, s in enumerate(path):
            p *= probs[t][s]
        collapsed = collapse(path)
        path_data.append((path, collapsed, p))
        print(f"  {str(path):>15}  {collapsed:>10}  {p:>10.6f}")

    valid_a = [(pa, co, pr) for pa, co, pr in path_data if co == "A"]
    total_a = sum(pr for _, _, pr in valid_a)

    print(f"\nPaths collapsing to 'A':")
    for pa, co, pr in sorted(valid_a, key=lambda x: -x[2]):
        print(f"  {str(pa):>15}  ->  {co}  prob={pr:.6f}")

    print(f"\nP(target='A') = sum of valid path probs = {total_a:.6f}")

    best = max(path_data, key=lambda x: x[2])
    print(f"\nBest single path: {best[0]} -> '{best[1]}' with prob={best[2]:.6f}")
    print(f"P(target='A') = {total_a:.6f} > best single path prob = {best[2]:.6f}")
    print(f"=> CTC sums over ALL valid alignments, not just the best one.")


if __name__ == "__main__":
    main()
