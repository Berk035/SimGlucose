from simglucose.analysis.report import report
import pandas as pd
from pathlib import Path


# get the path to the example folder
exmaple_pth = Path(__file__).parent

# find all csv with pattern *#*.csv, e.g. adolescent#001.csv
result_filenames = list(exmaple_pth.glob(
    'results/*#*.csv'))
patient_names = [f.stem for f in result_filenames]
df = pd.concat(
        [pd.read_csv(str(f), index_col=0) for f in result_filenames],
        keys=patient_names)
report(df)
