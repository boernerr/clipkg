'''The purpose of this file is to take a raw SRSP extract fron Oracle in .csv format.
This file  normalizes the text by lemmatizing and tokenizing text. '''
import os
import re
import glob

from datetime import datetime
from os import path
from sklearn.pipeline import Pipeline
from time import perf_counter

from clipkg import DATA_PATH, MODEL_OUT_PATH
from clipkg.src.transformers import (FormatEstimatorRussia, Word2VecNormalizer, FormatEstimator)


class BadInputFileError(Exception):
    """To allow for alternate attempts at getting input files."""
    def __init__(self):
        pass


def main():
    while True:
        try:
            file_path = get_input_file_from_user()
        except BadInputFileError:
            continue
        else:
            break
    print(f"Starting time of run: {datetime.now().strftime('%H:%M:%S')}")
    print(f'Received a valid file path: {file_path} \nThis may take upwards of 30 minutes, please wait...')
    file_name = path.basename(file_path)
    data_src = {'SRSP': {'text_col': 'MEMINCIDENTREPORT',
                         'output_cols': ['TXTSSN', 'EFOLDERID', 'TXTINCIDENTTYPE', 'MEMINCIDENTREPORT']},
                'OTHER': {'text_col': 'ALLEGATION',
                          'output_cols': ['SSN', 'ALLEGATION_ID', 'EVENT', 'ALLEGATION']}
                }
    # Determine which text_col based on the input file should be used for the pipeline below:
    if re.search('.*SRSP.', file_name):
        table_src = 'SRSP'
        # The column to perform analysis on:
        text = data_src.get('SRSP').get('text_col')
    else:
        table_src = 'OTHER'
        # The column to perform analysis on:
        text = data_src.get('OTHER').get('text_col')

    # This step is universal regardless of data source used:
    pipeline_steps = [
        ('normalize', Word2VecNormalizer(text_col=text, return_type=None))
    ]
    # Depending on what data source is used, we use either of the FormatEstimator variations below; we pre-pend this step
    # to the pipeline_steps above:
    if table_src == 'SRSP':
        pipeline_steps.insert(0, ('format_estimator', FormatEstimator(text_col=text, path_=file_path)))
    elif table_src == 'OTHER':
        pipeline_steps.insert(0, ('format_estimator', FormatEstimatorRussia(text_col=text, path_=file_path)))
        # Need to add in CondenseTransformer() in order to handle duplicates in the Russia subset! See russia_analysis.py for details!

    pipe_main = Pipeline(pipeline_steps)
    # Create our data:
    t0 = perf_counter()
    formatter = FormatEstimator(text_col=text, path_=file_path)
    formatter._create_df()
    df = formatter.df
    # Return text column as list of tokens with stop words removed:
    df = pipe_main.fit_transform(df)
    # Only preserve these columns:
    df = df[data_src.get(table_src).get('output_cols')]
    # To .csv here:
    today = datetime.today().strftime('%m_%d_%Y')
    file_name = path.splitext(path.basename(file_path))[0]# Think this can be reduced
    out_path = path.join(MODEL_OUT_PATH, f'{file_name}_format_for_nn_{today}.csv')
    df.to_csv(out_path, index=False)
    # color = '\033[96m' # colors for text in terminal
    # end_color = '\033[0m'
    # This should go to a logging file:
    print(f'CREATED : {out_path}')
    print(f'Time to execute: {round((perf_counter() - t0)/60, 4)} minutes')
    # Returning the created file so I can feed the file to the next step down the pipeline:
    return path.basename(out_path), text


def get_input_file_from_user() -> str:
    """Get a valid path reference from user input.

    First assume a full path, then assume a local repo file.

    Raises:
        BadInputFileError: If input is neither a repo data file or a full path on disk.

    """
    local_repo_raw_data_dir = path.join(DATA_PATH, 'raw')
    local_repo_raw_data_files = path.join(local_repo_raw_data_dir, r'*.csv')
    # glob is case-insensitive by default, this will catch both .csv and .CSV
    print(f'Input files available in the default directory ({local_repo_raw_data_dir}):')
    print('-' * 50, '\n'.join([x for x in glob.glob(local_repo_raw_data_files)]), '-' * 50, sep='\n')
    input_prompt = 'Input one of the filenames above or the full path to another input file: '
    file_path = path.normpath(input(input_prompt))

    try:
        os.stat(file_path)
    except FileNotFoundError:
        local_repo_raw_data_path = path.join(local_repo_raw_data_dir, file_path)
        try:
            os.stat(local_repo_raw_data_path)
        except FileNotFoundError:
            # If possible, list the contents of the supplied dir and raise error to try again
            file_path_parent = path.dirname(file_path)
            if path.isdir(file_path_parent):
                print('Perhaps you meant one of these?')
                csv_glob_in_parent = path.join(file_path_parent, '*.csv')
                for basename in glob.glob(csv_glob_in_parent):
                    print('\t', basename)
            raise BadInputFileError
        else:
            file_path = local_repo_raw_data_path
    return file_path


if __name__ == '__main__':
    main()
