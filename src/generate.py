from io import UnsupportedOperation
import os
import sys
import pandas as pd
import argparse
import yaml

def load_from_table():
    # df = pd.read_html(table_text)[0]
    with open('freq_table.html', 'r') as f:
        data = f.read()
        df = pd.read_html(data)[0]
    df.columns = df.columns.get_level_values(1)
    df = df[['Scientific name[5]', 'Frequency (Hz) (Equal temperament) [6]']]
    df.columns = ['name', 'freq']
    return df

def generate_random_tone():
    raise UnsupportedOperation

if __name__ == "__main__":
    
    params = yaml.safe_load(open('params.yaml'))['generate']

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython generate.py out_folder\n")
        sys.exit(1)

    out_folder = os.path.abspath(sys.argv[1])
    os.makedirs(out_folder, exist_ok=True)
    freq_file = os.path.join(out_folder, 'freq.csv')
    raw_file = os.path.join(out_folder, 'raw.csv')

    df = load_from_table()
    df.to_csv(freq_file)

    generate_random_tone()



    