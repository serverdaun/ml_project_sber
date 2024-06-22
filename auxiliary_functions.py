import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder


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
    def categorical_feature_te(df, column, target_feature='CR') -> pd.DataFrame:
        te = TargetEncoder()

        df[f'{column}_te'] = te.fit_transform(df[column], df[target_feature])
        df = df.drop(columns=[column], axis=1)

        return df

    @staticmethod
    def numerical_feature_std(df, column) -> pd.DataFrame:
        std = StandardScaler()

        std.fit(df[[column]])
        std_column = std.transform(df[[column]])

        df[f'{column}_std'] = std_column
        df = df.drop(columns=[column], axis=1)

        return df

    @staticmethod
    def evaluate_model(y_true, y_pred, y_pred_proba):
        result = {}

        result['accuracy'] = accuracy_score(y_true, y_pred)

        result['f1_score'] = f1_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        result['confusion_matrix'] = cm

        result['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        print(f"Accuracy: {result['accuracy']:.2f}")
        print(f"F1 Score: {result['f1_score']:.2f}")
        print(f"ROC-AUC Score: {result['roc_auc']:.2f}")

        return result

    @staticmethod
    def create_roc_auc_curve(roc_auc, y_test, y_pred_proba):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        plt.show()
