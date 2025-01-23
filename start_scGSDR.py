import subprocess

refer_dataset_path = "./Data/example_refer.csv"
query_dataset_path = "./Data/example_query.csv"

subprocess.run(f"python ./scGSDR.py --refer_dataset_path {refer_dataset_path} --query_dataset_path {query_dataset_path}")

