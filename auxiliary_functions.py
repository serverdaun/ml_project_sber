import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


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

    @staticmethod
    def categorical_feature_ohe(df, column) -> pd.DataFrame:
        ohe = OneHotEncoder(sparse_output=False)

        ohe.fit(df[[column]])
        ohe_columns = ohe.transform(df[[column]])
        ohe_df = pd.DataFrame(ohe_columns, columns=ohe.get_feature_names_out([column]))

        df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        df = df.drop(columns=[column], axis=1)

        return df

    @staticmethod
    def numerical_feature_std(df, column) -> pd.DataFrame:
        std = StandardScaler()

        std.fit(df[[column]])
        std_column = std.transform(df[[column]])
        ohe_df = pd.DataFrame(std_column, columns=std_column.get_feature_names_out([column]))

        df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        df = df.drop(columns=[column], axis=1)

        return df
