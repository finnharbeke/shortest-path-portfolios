import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rp = pd.read_csv('./cache/random_pairs/sh_portfolios.csv', index_col=0)
    # rp = pd.read_csv('./cache/sh_portfolios.csv', index_col=0)
    rp['improve'] = rp['factor'] - 1
    sns.lineplot(rp, x='k', y='improve', hue='method')
    plt.gca().set_yscale('log')
    plt.savefig('greedy_vs_mf.png')
    

