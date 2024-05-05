import pandas as pd


class PreprocessingUtils:

    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> None:
        duplicates = df[df.duplicated(keep=False)]
        if duplicates.empty:
            print('No duplicate rows.')
        else:
            print(f'Number of duplicated rows: {len(duplicates)}')
            print(f'Duplicated rows:\n {duplicates}')

    @staticmethod
    def print_basic_stats(df: pd.DataFrame, level='all') -> None:
        if level == 'all':
            print(f'Number of rows: {df.shape[0]}')
            print(f'Number of columns: {df.shape[1]}')
            print(df.describe(include='all'))
            print(df.info())
        elif level == 'shape':
            print(f'Number of rows: {df.shape[0]}')
            print(f'Number of columns: {df.shape[1]}')

    @staticmethod
    def missing_values_percentage(df: pd.DataFrame) -> None:
        missing_values_df_level = (df.notna().all(axis=1).sum() / df.shape[0]).round(2)
        print(f'Complete rows in percentage: {missing_values_df_level}')
        missing_values = ((df.isna().sum() / df.shape[0]) * 100).sort_values(ascending=False).round(2)
        print(f'Missing values in percentage:\n{missing_values[missing_values > 0]}')
