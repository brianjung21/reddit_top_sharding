import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('../data/reddit_brand_harvest_hot.csv')
    brands = df['brand'].unique()
    df1 = df[df['brand'] == df['alias_used']]
    df1.to_csv('../data/hot_samples.csv', index=False)
