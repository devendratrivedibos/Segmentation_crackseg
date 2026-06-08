import re
import re

def top10_metric(log_file, metric_name):
    results = []

    with open(log_file, "r") as f:
        text = f.read()

    # Accuracy is stored as a single value
    if metric_name.lower() == "accuracy":

        epoch_blocks = re.findall(
            r"\[epoch:\s*(\d+)\](.*?)(?=\[epoch:|\Z)",text,re.DOTALL)

        for epoch, block in epoch_blocks:
            match = re.search(r"accuracy:\s*([\d.]+)", block)
            if match:
                results.append((int(epoch), float(match.group(1))))

    else:
        blocks = re.findall(
            rf"{metric_name}:\s*\[(.*?)\]",text,re.DOTALL | re.IGNORECASE)

        for epoch_idx, block in enumerate(blocks):
            values = [float(x) for x in re.findall(r"[\d.]+", block)]
            first6 = values[1:6]
            avg_first6 = sum(first6) / len(first6) if first6 else 0
            results.append((epoch_idx, avg_first6))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== Top 10 Epochs by {metric_name.upper()} ===")
    for i, (epoch, value) in enumerate(results[:20], start=1):
        print(f"{i}. Epoch {epoch} → {value:.2f}")
# --- Usage ---
log_file = r"D:\Devendra_Files\segmentation_training\weights\UNet_resnet\6junetpp_resnet_b4-results.txt"
top10_metric(log_file, "recall")
top10_metric(log_file, "precision")
top10_metric(log_file, "accuracy")
