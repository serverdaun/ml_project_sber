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


def filter_sessions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Removes unnecessary columns from a sessions DataFrame"""
    df_filtered = df.copy()
    columns_to_drop = [
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'visit_number',
        'device_model'
    ]
    return df_filtered.drop(columns_to_drop, axis=1)


def fillna_utm_source(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df.copy()
    df_filtered.utm_source = df_filtered.utm_source.fillna('(not set')
    return df_filtered


def main():

    # Preprocess file with hits to receive a list of sessions with target actions.
    hits_df_pipeline = Pipeline(steps=[
        ('filter_hits_df', FunctionTransformer(filter_hits_df)),
        ('get_unique_hits_cr', FunctionTransformer(get_unique_hits_cr))
    ])

    unique_hits_cr = hits_df_pipeline.fit_transform(pd.read_csv(filepath_or_buffer=HITS_PATH))

    # Add CR flags to sessions DataFrame
    sessions_df = pd.read_csv(filepath_or_buffer=SESSIONS_PATH)
    df = pd.merge(sessions_df, unique_hits_cr, on='session_id', how='inner')

    x = df.drop(['CR'], axis=1)
    y = df['CR']

    # Separate features for categorical (low and high cardinality) and numerical
    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object'])

    unique_value_counts = df[categorical_features].nunique()
    low_cardinality_cat_features = unique_value_counts[unique_value_counts <= 400].index.tolist()
    high_cardinality_cat_features = unique_value_counts[unique_value_counts > 400].index.tolist()

    preprocessor = Pipeline(steps=[
        ('filter_sessions_df', FunctionTransformer(filter_sessions_df)),
        ('fillna_utm_source', FunctionTransformer(fillna_utm_source))
    ])


if __name__ == '__main__':
    main()
