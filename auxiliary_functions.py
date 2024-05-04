class PreprocessingUtils:

    @staticmethod
    def check_duplicates(df):
        duplicates = df[df.duplicated(keep=False)]
        if duplicates.empty:
            print('No duplicate rows.')
        else:
            print(f'Number of duplicated rows: {len(duplicates)}')
            print(f'Duplicated rows:\n {duplicates}')
