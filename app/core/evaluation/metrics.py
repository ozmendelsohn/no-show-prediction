from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_curve
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import mlflow
from app.utils import get_logger
from sklearn.preprocessing import LabelEncoder

logger = get_logger(__name__)

class ModelEvaluator:
    """Class for model evaluation and visualization."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing evaluation parameters
        """
        self.config = config
        self.metrics = {}
        self.figures = {}
        
        # Time series cross-validation settings
        self.n_splits = config.get('n_splits', 5)
        self.gap = config.get('gap', 0)  # Gap between train and test sets
        self.test_size = config.get('test_size', None)  # Size of the test set
    
    def create_time_series_splits(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        n_splits: int = 5,
        gap: int = 7,
        test_size: int = 30
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series cross-validation splits.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        date_column : str, optional
            Name of the date column, by default 'date'
        n_splits : int, optional
            Number of splits, by default 5
        gap : int, optional
            Gap between train and test in days, by default 7
        test_size : int, optional
            Size of test set in days, by default 30

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_idx, test_idx) tuples
        """
        if date_column not in data.columns:
            raise ValueError(f"Date column {date_column} not found in data")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Sort by date
        sorted_idx = data[date_column].sort_values().index
        data = data.loc[sorted_idx].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Get unique dates
        unique_dates = pd.Series(data[date_column].unique()).sort_values()
        
        # Calculate default test size if not provided
        if self.test_size is None:
            self.test_size = len(unique_dates) // (self.n_splits + 1)
        
        # Calculate gap in days if not provided
        if self.gap == 0:
            # Default gap is 7 days
            self.gap = pd.Timedelta(days=7)
        elif isinstance(self.gap, (int, float)):
            self.gap = pd.Timedelta(days=self.gap)
        
        splits = []
        test_end_idx = len(data)
        
        for i in range(self.n_splits):
            # Calculate test start and end dates
            test_end_date = data[date_column].iloc[test_end_idx - 1]
            test_start_date = test_end_date - pd.Timedelta(days=self.test_size)
            train_end_date = test_start_date - self.gap
            
            # Get indices for this split
            test_mask = (data[date_column] >= test_start_date) & (data[date_column] <= test_end_date)
            train_mask = data[date_column] <= train_end_date
            
            test_idx = data[test_mask].index
            train_idx = data[train_mask].index
            
            if len(test_idx) == 0 or len(train_idx) == 0:
                logger.warning(f"Split {i} has empty train or test set, skipping")
                continue
            
            splits.append((train_idx, test_idx))
            
            # Update test_end_idx for next iteration
            test_end_idx = data[data[date_column] <= test_start_date].index[-1]
            
            # Break if we don't have enough data for another split
            if len(train_idx) < len(test_idx):
                break
        
        if not splits:
            raise ValueError("No valid splits could be created. Check your data and split parameters.")
        
        # Reverse splits so they go forward in time
        splits = splits[::-1]
        
        # Plot splits visualization
        self._plot_time_series_splits(data[date_column], splits)
        
        return splits
    
    def _plot_time_series_splits(
        self,
        dates: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Visualize time series splits.

        Parameters
        ----------
        dates : pd.Series
            Series of dates
        splits : List[Tuple[np.ndarray, np.ndarray]]
            List of (train_idx, test_idx) for each split
        """
        plt.figure(figsize=(15, self.n_splits * 1.5))
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Plot training samples
            train_dates = dates[train_idx]
            plt.plot(train_dates, [i] * len(train_dates), 'b.', label='Training' if i == 0 else "")
            
            # Plot gap
            if len(train_idx) > 0 and len(test_idx) > 0:
                last_train_date = dates[train_idx].max()
                first_test_date = dates[test_idx].min()
                gap_dates = pd.date_range(last_train_date, first_test_date, freq='D')
                plt.plot(gap_dates, [i] * len(gap_dates), 'g.', label='Gap' if i == 0 else "")
            
            # Plot testing samples
            test_dates = dates[test_idx]
            plt.plot(test_dates, [i] * len(test_dates), 'r.', label='Testing' if i == 0 else "")
        
        plt.ylim(-0.5, len(splits) - 0.5)
        plt.xlabel('Date')
        plt.ylabel('CV Split')
        plt.title('Time Series Cross-Validation Splits')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self.figures['time_series_splits'] = plt.gcf()
        plt.close()
    
    def evaluate_time_series(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        date_column: str,
        categorical_features: List[str] = None
    ) -> Dict[str, List[float]]:
        """
        Evaluate model using time series cross-validation.

        Parameters
        ----------
        model : Any
            Trained model object
        X : pd.DataFrame
            Feature DataFrame
        y : pd.Series
            Target series
        date_column : str
            Name of the date column for sorting
        categorical_features : List[str], optional
            List of categorical feature names, by default None

        Returns
        -------
        Dict[str, List[float]]
            Dictionary of metric lists for each split
        """
        splits = self.create_time_series_splits(X, y, date_column)
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        # Initialize label encoders dictionary
        encoders = {}
        
        # Identify categorical columns if not provided
        if categorical_features is None:
            categorical_features = X.select_dtypes(
                include=['category', 'object', 'bool']
            ).columns.tolist()
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Get split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create copies for encoding
            X_train_encoded = X_train.copy()
            X_test_encoded = X_test.copy()
            
            # Store datetime columns
            datetime_cols = X_train_encoded.select_dtypes(include=['datetime64']).columns
            
            # Drop datetime columns before encoding
            X_train_encoded = X_train_encoded.drop(columns=datetime_cols)
            X_test_encoded = X_test_encoded.drop(columns=datetime_cols)
            
            # Encode categorical features
            for col in categorical_features:
                if col in X_train_encoded.columns:
                    if col not in encoders:
                        encoders[col] = LabelEncoder()
                        X_train_encoded[col] = encoders[col].fit_transform(X_train_encoded[col].astype(str))
                    else:
                        X_train_encoded[col] = encoders[col].transform(X_train_encoded[col].astype(str))
                    
                    # Handle unseen categories in test set
                    X_test_encoded[col] = X_test_encoded[col].astype(str).map(
                        lambda x: -1 if x not in encoders[col].classes_ else encoders[col].transform([x])[0]
                    )
            
            # Convert to float32 for better memory usage
            X_train_encoded = X_train_encoded.astype(np.float32)
            X_test_encoded = X_test_encoded.astype(np.float32)
            
            # Ensure feature names are preserved
            X_train_encoded = pd.DataFrame(X_train_encoded, columns=X_train_encoded.columns)
            X_test_encoded = pd.DataFrame(X_test_encoded, columns=X_test_encoded.columns)
            
            # Train model on this split
            model.fit(X_train_encoded, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_encoded)
            y_prob = model.predict_proba(X_test_encoded)[:, 1]
            
            # Calculate metrics
            split_metrics = self.evaluate(y_test, y_pred, y_prob, split=f"split_{i}")
            
            # Store metrics
            for metric in cv_metrics.keys():
                if metric in split_metrics:
                    cv_metrics[metric].append(split_metrics[metric])
            
            # Log split details
            logger.info(f"\nSplit {i+1} Results:")
            logger.info(f"Train size: {len(X_train_encoded)}, Test size: {len(X_test_encoded)}")
            logger.info(f"Train period: {X_train[date_column].min()} to {X_train[date_column].max()}")
            logger.info(f"Test period: {X_test[date_column].min()} to {X_test[date_column].max()}")
            for metric, value in split_metrics.items():
                if isinstance(value, (np.ndarray, list)):
                    logger.info(f"{metric}: {np.mean(value):.4f}")
                else:
                    logger.info(f"{metric}: {value:.4f}")
        
        # Calculate and log average metrics
        logger.info("\nOverall Cross-validation Results:")
        for metric, values in cv_metrics.items():
            mean_val = float(np.mean(values))  # Convert to float to avoid numpy type issues
            std_val = float(np.std(values))
            logger.info(f"{metric}: {mean_val:.4f} (+/- {std_val:.4f})")
            mlflow.log_metric(f'cv_avg_{metric}', mean_val)
            mlflow.log_metric(f'cv_std_{metric}', std_val)
        
        # Plot metrics distribution
        self._plot_cv_metrics_distribution(cv_metrics)
        
        return cv_metrics
    
    def _plot_cv_metrics_distribution(
        self,
        cv_metrics: Dict[str, List[float]]
    ) -> None:
        """
        Plot distribution of metrics across CV splits.

        Parameters
        ----------
        cv_metrics : Dict[str, List[float]]
            Dictionary of metric lists for each split
        """
        plt.figure(figsize=(12, 6))
        
        # Create violin plots for better distribution visualization
        data = []
        labels = []
        for metric, values in cv_metrics.items():
            data.append(values)
            labels.append(metric)
        
        # Plot violin plot with individual points
        parts = plt.violinplot(data, showmeans=True)
        
        # Customize violin plot colors
        for pc in parts['bodies']:
            pc.set_facecolor('#2196F3')
            pc.set_alpha(0.7)
        
        # Add individual points
        for i, d in enumerate(data):
            plt.scatter([i + 1] * len(d), d, c='red', alpha=0.5, s=20)
        
        # Customize plot
        plt.title('Cross-validation Metrics Distribution', fontsize=12, pad=20)
        plt.ylabel('Score', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
        
        # Add mean ± std annotations
        for i, (metric, values) in enumerate(cv_metrics.items()):
            mean = np.mean(values)
            std = np.std(values)
            plt.annotate(f'μ={mean:.3f}\nσ={std:.3f}',
                        xy=(i + 1, mean),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        if hasattr(self, 'artifact_path'):
            plot_path = Path(self.artifact_path) / "cv_metrics_distribution.png"
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            mlflow.log_artifact(str(plot_path))
        
        self.figures['cv_metrics_distribution'] = plt.gcf()
        plt.close()
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        split: str = "train"
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_prob : Optional[np.ndarray], optional
            Predicted probabilities, by default None
        split : str, optional
            Data split name, by default "train"

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, average='binary'))
        metrics['recall'] = float(recall_score(y_true, y_pred, average='binary'))
        metrics['f1'] = float(f1_score(y_true, y_pred, average='binary'))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = float(cm.mean())
        metrics['tn'] = float(cm[0, 0])
        metrics['fp'] = float(cm[0, 1])
        metrics['fn'] = float(cm[1, 0])
        metrics['tp'] = float(cm[1, 1])
        
        # ROC AUC and PR AUC if probabilities are available
        if y_prob is not None:
            if len(y_prob.shape) > 1:
                y_prob = y_prob[:, 1]
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
                metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        # Store metrics
        self.metrics[split] = metrics
        
        # Log metrics to MLflow
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{split}_{name}", value)
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        split: str = "train",
        normalize: bool = True,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot confusion matrix.

        Parameters
        ----------
        split : str, optional
            Data split name, by default "train"
        normalize : bool, optional
            Whether to normalize the confusion matrix, by default True
        save_path : Optional[Path], optional
            Path to save the plot, by default None
        """
        if split not in self.metrics:
            logger.warning(f"No metrics found for split: {split}")
            return
            
        cm = self.metrics[split]['confusion_matrix']
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2%' if normalize else 'd',
            cmap='Blues',
            xticklabels=['No Show', 'Show'],
            yticklabels=['No Show', 'Show']
        )
        plt.title(f'Confusion Matrix ({split})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path / f"confusion_matrix_{split}.png")
            mlflow.log_artifact(str(save_path / f"confusion_matrix_{split}.png"))
        
        self.figures[f'confusion_matrix_{split}'] = plt.gcf()
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: Dict[str, np.ndarray],
        y_prob: Dict[str, np.ndarray],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot ROC curves for multiple splits.

        Parameters
        ----------
        y_true : Dict[str, np.ndarray]
            Dictionary of true labels for each split
        y_prob : Dict[str, np.ndarray]
            Dictionary of predicted probabilities for each split
        save_path : Optional[Path], optional
            Path to save the plot, by default None
        """
        plt.figure(figsize=(8, 6))
        
        for split, y_true_split in y_true.items():
            y_prob_split = y_prob[split]
            if len(y_prob_split.shape) > 1:
                y_prob_split = y_prob_split[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true_split, y_prob_split)
            roc_auc = roc_auc_score(y_true_split, y_prob_split)
            
            plt.plot(
                fpr, tpr,
                label=f'{split} (AUC = {roc_auc:.3f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path / "roc_curve.png")
            mlflow.log_artifact(str(save_path / "roc_curve.png"))
        
        self.figures['roc_curve'] = plt.gcf()
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: Dict[str, np.ndarray],
        y_prob: Dict[str, np.ndarray],
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot Precision-Recall curves for multiple splits.

        Parameters
        ----------
        y_true : Dict[str, np.ndarray]
            Dictionary of true labels for each split
        y_prob : Dict[str, np.ndarray]
            Dictionary of predicted probabilities for each split
        save_path : Optional[Path], optional
            Path to save the plot, by default None
        """
        plt.figure(figsize=(8, 6))
        
        for split, y_true_split in y_true.items():
            y_prob_split = y_prob[split]
            if len(y_prob_split.shape) > 1:
                y_prob_split = y_prob_split[:, 1]
            
            precision, recall, _ = precision_recall_curve(y_true_split, y_prob_split)
            pr_auc = average_precision_score(y_true_split, y_prob_split)
            
            plt.plot(
                recall, precision,
                label=f'{split} (AP = {pr_auc:.3f})'
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path / "pr_curve.png")
            mlflow.log_artifact(str(save_path / "pr_curve.png"))
        
        self.figures['pr_curve'] = plt.gcf()
        plt.close()
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split: str = "train",
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate detailed classification report.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        split : str, optional
            Data split name, by default "train"
        save_path : Optional[Path], optional
            Path to save the report, by default None

        Returns
        -------
        str
            Classification report
        """
        report = classification_report(y_true, y_pred)
        
        if save_path:
            report_path = save_path / f"classification_report_{split}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            mlflow.log_artifact(str(report_path))
        
        return report
    
    def save_all_plots(self, save_path: Path) -> None:
        """
        Save all generated plots.

        Parameters
        ----------
        save_path : Path
            Path to save the plots
        """
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in self.figures.items():
            plot_path = save_path / f"{name}.png"
            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
            mlflow.log_artifact(str(plot_path))
            
            # Create interactive HTML version using plotly
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                if 'confusion_matrix' in name:
                    fig_plotly = self._convert_confusion_matrix_to_plotly(fig)
                elif 'roc_curve' in name:
                    fig_plotly = self._convert_roc_curve_to_plotly(fig)
                elif 'feature_importance' in name:
                    fig_plotly = self._convert_feature_importance_to_plotly(fig)
                else:
                    continue
                
                html_path = save_path / f"{name}.html"
                fig_plotly.write_html(str(html_path))
                mlflow.log_artifact(str(html_path))
            except Exception as e:
                logger.warning(f"Could not create interactive plot for {name}: {str(e)}")
    
    def _convert_confusion_matrix_to_plotly(self, fig):
        """Convert matplotlib confusion matrix to plotly figure."""
        import plotly.graph_objects as go
        
        # Extract data from matplotlib figure
        ax = fig.axes[0]
        im = ax.get_images()[0]
        data = im.get_array()
        
        # Create plotly figure
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=['No Show', 'Show'],
            y=['No Show', 'Show'],
            text=data,
            texttemplate='%{text:.0f}',
            textfont={'size': 20},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        return fig
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get summary of all metrics as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Summary of metrics
        """
        summary = []
        for split, metrics in self.metrics.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    summary.append({
                        'split': split,
                        'metric': metric,
                        'value': value
                    })
        
        return pd.DataFrame(summary)
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        importance_type: str = "native",
        n_top_features: int = 20,
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Plot feature importance.

        Parameters
        ----------
        model : Any
            Trained model object
        feature_names : List[str]
            List of feature names
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        importance_type : str, optional
            Type of importance to plot ('native' or 'permutation'), by default "native"
        n_top_features : int, optional
            Number of top features to display, by default 20
        save_path : Optional[Path], optional
            Path to save the plot, by default None

        Returns
        -------
        pd.DataFrame
            Feature importance DataFrame
        """
        importances = None
        std = None
        
        try:
            if importance_type == "native":
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    std = np.zeros_like(importances)
                else:
                    logger.warning("Model doesn't have native feature importance. Using permutation importance instead.")
                    importance_type = "permutation"
            
            if importance_type == "permutation":
                # Ensure X is 2D array
                if len(X.shape) == 1:
                    X = X.values.reshape(-1, 1)
                
                result = permutation_importance(
                    model, X, y,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
                importances = result.importances_mean
                std = result.importances_std
            
            if importances is None:
                logger.warning("Could not calculate feature importance")
                return pd.DataFrame()
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'std': std
            })
            
            # Sort by importance and get top N features
            feature_importance = feature_importance.sort_values(
                'importance', ascending=False
            ).head(n_top_features)
            
            # Plot horizontal bar chart
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=feature_importance,
                y='feature',
                x='importance',
                palette='viridis'
            )
            
            plt.title(f'Top {n_top_features} Feature Importance ({importance_type.title()})')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            
            if save_path:
                save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path / f"feature_importance_{importance_type}.png", bbox_inches='tight')
                mlflow.log_artifact(str(save_path / f"feature_importance_{importance_type}.png"))
            
            self.figures[f'feature_importance_{importance_type}'] = plt.gcf()
            plt.close()
            
            # Log feature importance values
            importance_dict = dict(zip(feature_importance['feature'], feature_importance['importance']))
            for feature, importance in importance_dict.items():
                mlflow.log_metric(f"feature_importance_{feature}", float(importance))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame()
    