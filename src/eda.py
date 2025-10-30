import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_distribution(df, feature):
    plt.figure(figsize=(8,5))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()