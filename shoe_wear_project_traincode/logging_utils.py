# logging_utils.py

import os
import csv
import json

def append_log_csv(log_path, row_dict, fieldnames):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def save_confusion_matrix_txt(path, cm, info=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        if info is not None:
            f.write("Additional Info:\n")
            f.write(json.dumps(info, indent=2, ensure_ascii=False))
