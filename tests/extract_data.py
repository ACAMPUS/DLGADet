
path=r'D:\dataset\实验记录\20-using-origin-dataset-tt100k-batch32-add-detect-modify\log.txt'
import re
import re

# Initialize lists to store the metrics
all_P = []
all_R = []
all_mAP50 = []
all_mAP5095 = []

# Open the log file
with open(path, 'r') as file:
    # Loop through each line in the file
    for line in file:

        # If the line contains the desired metrics, extract them using regular expressions
        if re.search(r"Class\s+Images\s+Labels\s+P\s+R\s+mAP@.5\s+mAP@.5:.95", line):
            metrics = re.findall(r"\d+\.\d+", line)
            all_P.append(float(metrics[2]))
            all_R.append(float(metrics[3]))
            all_mAP50.append(float(metrics[4]))
            all_mAP5095.append(float(metrics[5]))

# Print the metrics for each epoch
for i in range(len(all_P)):
    print(f"Epoch {i + 1}: P={all_P[i]}, R={all_R[i]}, mAP@.5={all_mAP50[i]}, mAP@.5:.95={all_mAP5095[i]}")



