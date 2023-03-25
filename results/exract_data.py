import re

# Initialize lists to store the metrics
all_P = []
all_R = []
all_mAP50 = []
all_mAP5095 = []
path=r'D:\dataset\实验记录\47\log.txt'

# Open the log file
with open(path, 'r') as file:
    # Loop through each line in the file
    for line in file:

        # If the line contains the desired metrics, extract them using regular expressions
        if re.search(r"all\s+3067\s+7698\s+", line):
            metrics = re.findall(r"\d+\.\d+", line)
            all_P.append(float(metrics[0]))
            all_R.append(float(metrics[1]))
            all_mAP50.append(float(metrics[2]))
            all_mAP5095.append(float(metrics[3]))

# Print the metrics for each epoch
for i in range(len(all_P)):
    print(f"Epoch {10*(i+1)}: P={all_P[i]}, R={all_R[i]}, mAP@.5={all_mAP50[i]}, mAP@.5:.95={all_mAP5095[i]}")

