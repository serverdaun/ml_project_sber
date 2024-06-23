import datetime
import dill
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from category_encoders import TargetEncoder

HITS_PATH = '../data/skillbox_diploma_main_dataset_sberautopodpiska/ga_hits-002.csv'
SESSIONS_PATH = '../data/skillbox_diploma_main_dataset_sberautopodpiska/ga_sessions.csv'
SOCIAL_MEDIA_SOURCES = [
    'QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
    'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
]
TOP_BROWSERS = ['chrome', 'safari', 'firefox']
POPULAR_BRANDS = ['samsung', 'apple', 'xiaomi', 'huawei']
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
    df_upd = df.copy()
    columns_to_drop = [
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'visit_number',
        'device_model'
    ]
    df_upd = df_upd.drop(columns_to_drop, axis=1)
    return df_upd


def fillna_device_os(df: pd.DataFrame) -> pd.DataFrame:
    # Fillna for Apple gadgets
    df_upd = df.copy()
    df_upd.loc[(df_upd.device_os.isna()) & (df_upd.device_brand == 'Apple'), 'device_os'] = 'iOS'

    # Fillna for Android based gadgets
    android_based = ['Samsung', 'Xiaomi', 'Huawei', 'Realme']
    df_upd.loc[
        (df_upd.device_os.isna()) & (df_upd.device_brand.isin(android_based)), 'device_os'] = 'Android'

    return df_upd


def is_organic(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd.loc[:, 'is_organic'] = df_upd['utm_medium'].apply(
        lambda x: 1 if x in ('organic', 'referral', '(none)') else 0)
    return df_upd


def in_app_browser(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd.loc[:, 'in_app_browser'] = df_upd.device_browser.apply(
        lambda x: 1 if x == 'safari (in-app)' or '.' in x else 0
    )
    return df_upd


def is_social_media_ad(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd.loc[:, 'is_social_media_ad'] = df_upd['utm_source'].apply(lambda x: 1 if x in SOCIAL_MEDIA_SOURCES else 0)
    return df_upd


def is_top_browser(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd.loc[:, 'is_top_browser'] = df_upd.device_browser.apply(lambda x: 1 if x in TOP_BROWSERS else 0)
    return df_upd


def is_popular_brand(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd.loc[:, 'is_popular_brand'] = df_upd.device_brand.apply(lambda x: 1 if x in POPULAR_BRANDS else 0)
    return df_upd


def create_screen_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    df_upd = df.copy()
    df_upd.loc[:, 'screen_width'] = df_upd.device_screen_resolution.apply(lambda x: x.split('x')[0]).astype('int')
    df_upd.loc[:, 'screen_height'] = df_upd.device_screen_resolution.apply(lambda x: x.split('x')[1]).astype('int')
    # df_upd = df_upd.drop(columns=['device_screen_resolution'], axis=1)
    return df_upd


def main():
    # Preprocess file with hits to receive a list of sessions with target actions.
    hits_df_pipeline = Pipeline(steps=[
        ('filter_hits_df', FunctionTransformer(filter_hits_df)),
        ('get_unique_hits_cr', FunctionTransformer(get_unique_hits_cr))
    ])

    unique_hits_cr = hits_df_pipeline.fit_transform(pd.read_csv(filepath_or_buffer=HITS_PATH, dtype=str))

    # Add CR flags to sessions DataFrame
    sessions_df = pd.read_csv(filepath_or_buffer=SESSIONS_PATH, dtype=str)
    df = pd.merge(sessions_df, unique_hits_cr, on='session_id', how='inner')
    df = filter_sessions_df(df)

    x = df.drop(['CR'], axis=1)
    y = df['CR']

    sample_weights = np.ones(len(y))
    sample_weights[y == 1] = 25

    # Separate features for categorical (low and high cardinality) and numerical
    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object'])

    unique_value_counts = df[categorical_features].nunique()
    low_cardinality_cat_features = unique_value_counts[unique_value_counts <= 400].index.tolist()
    high_cardinality_cat_features = unique_value_counts[unique_value_counts > 400].index.tolist()

    # Create pipelines for the final process
    preprocessor = Pipeline(steps=[
        ('fillna_device_os', FunctionTransformer(fillna_device_os)),
        ('is_organic', FunctionTransformer(is_organic)),
        ('is_social_media_ad', FunctionTransformer(is_social_media_ad)),
        ('in_app_browser', FunctionTransformer(in_app_browser)),
        ('is_top_browser', FunctionTransformer(is_top_browser)),
        ('is_popular_brand', FunctionTransformer(is_popular_brand)),
        ('create_screen_dimensions', FunctionTransformer(create_screen_dimensions))
    ])

    low_card_cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='(not set)')),
        ('ohe_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    high_card_cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='(not set)')),
        ('te_encoder', TargetEncoder())
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor2 = ColumnTransformer(transformers=[
        ('low_card_cat', low_card_cat_transformer, low_cardinality_cat_features),
        ('high_card_cat', high_card_cat_transformer, high_cardinality_cat_features),
        ('numerical_transform', numerical_transformer, numerical_features)
    ])

    model = GradientBoostingClassifier()

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('preprocessor2', preprocessor2),
        ('classifier', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = cross_val_score(pipe, x, y, cv=cv, scoring='roc_auc',
                                     params={'classifier__sample_weight': sample_weights})
    print(f'Mean ROC-AUC score: {roc_auc_scores.mean():.4f}')

    pipe.fit(x, y, classifier__sample_weight=sample_weights)
    with open('model.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Target event prediction model',
                'author': 'Vasilii Tokarev',
                'version': 1.0,
                'date': datetime.datetime.now(),
                'type': type(pipe.named_steps["classifier"]).__name__,
                'score': roc_auc_scores.mean().round(2),
            }
        }, file, recurse=True)


if __name__ == '__main__':
    main()
