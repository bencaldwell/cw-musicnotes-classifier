from io import UnsupportedOperation
import os
import sys
import pandas as pd
import argparse
import yaml
import numpy as np

def load_from_table():
    # df = pd.read_html(table_text)[0]
    df = pd.read_html('freq_table.html', encoding='utf-8')[0]
    df.columns = df.columns.get_level_values(1)
    df = df[['Scientific name[5]', 'Frequency (Hz) (Equal temperament) [6]']]
    df.columns = ['name', 'freq']
    df['name'] = df['name'].str.split('[ |/]', expand=True).iloc[:,0]
    return df

def generate_random_tones(df_freq, count=100):
    sample_freq = int(50e3) # a typical audio sample rate
    x = np.linspace(0, 1, sample_freq)
    rows = [] 
    for i in range(count):
        row = df_freq.sample()
        label = row.iloc[0]['name']
        freq = float(row.iloc[0]['freq'])
        y = np.sin(x * freq * 2 * np.pi)
        rows.append({'label': label, 'sample': y})
    df = pd.DataFrame(rows)
    return df

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

    df_freq = load_from_table()
    df_freq.to_csv(freq_file, index=False)

    df = generate_random_tones(df_freq)
    df.to_csv(raw_file, index=False)



    