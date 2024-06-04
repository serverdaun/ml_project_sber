import datetime
import dill
import pandas as pd
from auxiliary_functions import PreprocessingUtils
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.svm import SVC

HITS_PATH = '../data/skillbox_diploma_main_dataset_sberautopodpiska/ga_hits-002.csv'
SESSIONS_PATH = '../data/skillbox_diploma_main_dataset_sberautopodpiska/ga_sessions.csv'
TARGET_EVENTS = [
    'sub_car_claim_click', 'sub_car_claim_submit_click',
    'sub_open_dialog_click', 'sub_custom_question_submit_click',
    'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
    'sub_car_request_submit_click'
]


def filter_hits_df(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df.copy()
    columns_to_drop = [
        'hit_date',
        'hit_time',
        'hit_number',
        'hit_type',
        'hit_referer',
        'hit_page_path',
        'event_category',
        'event_label',
        'event_value'
    ]
    return df_filtered.drop(columns_to_drop, axis=1)


def get_unique_hits_cr(df: pd.DataFrame) -> pd.DataFrame:
    df['target_event'] = df.event_action.apply(lambda x: 1 if x in TARGET_EVENTS else 0)
    df['CR'] = df.groupby('session_id')['target_event'].transform('max').astype(int)
    unique_hits = df[['session_id', 'CR']].drop_duplicates(subset='session_id', keep='first')
    return unique_hits


def merge_sessions_w_hits(sessions_df: pd.DataFrame, hits_df: pd.DataFrame) -> pd.DataFrame:
    df = sessions_df.merge(hits_df, on='session_id', how='inner')
    return df


def main():

    hits_df_pipeline = Pipeline([
        ('filter_hits_df', FunctionTransformer(filter_hits_df)),
        ('get_unique_hits_cr', FunctionTransformer(get_unique_hits_cr))
    ])

    unique_hits_cr = hits_df_pipeline.fit_transform(pd.read_csv(filepath_or_buffer=HITS_PATH))

    # numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    # categorical_features = make_column_selector(dtype_include=['object'])
    #
    # unique_value_counts = df[categorical_features].nunique()
    # low_cardinality_cat_features = unique_value_counts[unique_value_counts <= 400].index.tolist()
    # high_cardinality_cat_features = unique_value_counts[unique_value_counts > 400].index.tolist()


if __name__ == '__main__':
    main()
