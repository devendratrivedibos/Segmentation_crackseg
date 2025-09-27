import re

def top10_metric(log_file, metric_name):
    results = []

    with open(log_file, "r") as f:
        text = f.read()

    # Find all blocks for given metric
    blocks = re.findall(rf"{metric_name}:\s*\[(.*?)\]", text, re.DOTALL)

    for epoch_idx, block in enumerate(blocks):
        values = [float(x) for x in re.findall(r"[\d.]+", block)]
        first6 = values[:6]
        avg_first6 = sum(first6) / len(first6) if first6 else 0
        results.append((epoch_idx, avg_first6))

    # Sort by avg_first6 in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== Top 10 Epochs by {metric_name.upper()} (AVG first 6) ===")
    for i, (epoch, avg) in enumerate(results[:10], start=1):
        print(f"{i}. Epoch {epoch} → {metric_name.upper()} AVG(first 6) = {avg:.2f}")


# --- Usage ---
log_file = r"D:\Devendra_Files\CrackSegFormer-main\weights\UNET_asphalt_4040\UNET_asp_4040-results.txt"
log_file = r"D:\Devendra_Files\CrackSegFormer-main\weights\26Sept_Asphalt_1024\26Sept_Asphalt_1024-results.txt"
# log_file = r"D:\Devendra_Files\CrackSegFormer-main\weights\25Sept_Asphalt_Augment\25Sept_Asphalt_Augment_-results.txt"
top10_metric(log_file, "recall")
top10_metric(log_file, "precision")
top10_metric(log_file, "accuracy")
