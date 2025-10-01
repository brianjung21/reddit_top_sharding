"""This is a script for transforming a dataset by pivoting specific columns and saving the results to a file.

The script reads a CSV input file containing beauty_daily counts of branded mentions, aggregates the counts into a pivot
table, and saves the processed data into a new CSV file. This script checks if the input file is empty or missing
specific required columns, handling these cases gracefully. The resulting pivot table aggregates values by date
and keyword while filling any missing values with 0. The output file is written in the same directory as the
input file.

Functions
---------
- make_pivot: Creates a pivot table from a pandas DataFrame over specified value columns.

"""

import pandas as pd
from pathlib import Path
from datetime import date, timedelta

INPUT_PATH = Path("../data/reddit_brand_harvest_top_month.csv")
OUT_POSTS = Path("../data/reddit_brand_pivoted.csv")


def make_pivot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Determine brand column
    if 'brand' in df.columns:
        brand_col = 'brand'
    elif 'keyword' in df.columns:
        brand_col = 'keyword'
    else:
        raise ValueError("Input is missing a brand/keyword column (expected 'brand' or 'keyword').")

    # Derive date column
    if 'created_iso' in df.columns:
        dt = pd.to_datetime(df['created_iso'], errors='coerce')
    elif 'created_utc' in df.columns:
        dt = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    else:
        raise ValueError("Input is missing created timestamp (expected 'created_iso' or 'created_utc').")
    df['date'] = dt.dt.date.astype('string')

    # Drop rows missing date or brand
    df = df.dropna(subset=['date', brand_col])

    # Each row is one mention
    df['mentions'] = 1

    # Pivot to date x brand counts
    return (
        df.pivot_table(
            index='date',
            columns=brand_col,
            values='mentions',
            aggfunc='sum',
            fill_value=0
        ).sort_index()
    )


def main():
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
    if df.empty:
        print('Input file is empty. Nothing to pivot.')
        return

    # Build pivot (sums aliases into brand totals per date)
    pivot_posts = make_pivot(df)

    # Save
    pivot_posts.to_csv(OUT_POSTS, encoding='utf-8-sig')

    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    if yesterday_str in pivot_posts.index.astype(str):
        today_df = pivot_posts.loc[[yesterday_str]]
        out_today = Path("../data") / f'brand_counts_{yesterday_str}.csv'
        today_df.to_csv(out_today, encoding='utf-8-sig')
        print('Done. Wrote:')
        print(f'  posts -> {OUT_POSTS.resolve()}')
        print(f'  yesterday -> {out_today.resolve()}')
    else:
        print('Done. Wrote:')
        print(f'  posts -> {OUT_POSTS.resolve()}')
        print(f'No data for yesterday ({yesterday_str}) found in pivot.')


if __name__ == '__main__':
    main()
