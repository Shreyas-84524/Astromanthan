"""Predict new objects with the saved astronomy pipeline, without retraining."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ML Model'))
from backend.engine.inference.unified_predictor import UnifiedPredictor
from backend.engine.astronomy.features import clean_numeric, RAW


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'new_predictions.csv')
    args=parser.parse_args()
    if args.output.resolve()==args.csv.resolve():
        raise ValueError('Output must not overwrite the input catalogue.')
    frame=pd.read_csv(args.csv,dtype=str)
    model=UnifiedPredictor.load(args.model_dir)
    p=model.predict_proba(frame)[:,1]
    result=pd.DataFrame({'source_row':np.arange(len(frame)),
        'predicted_class':np.where(p>=.5,'STAR','GALAXY'),
        'probability_STAR':p,'probability_GALAXY':1-p,
        'missing_or_invalid_inputs':clean_numeric(frame[RAW]).isna().sum(axis=1)})
    if 'Object_ID' in frame: result.insert(1,'Object_ID',frame.Object_ID)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Saved {len(result):,} binary predictions to {args.output}')
    print('Probabilities are model estimates. This binary model does not identify QSO.')


if __name__=='__main__':
    main()
