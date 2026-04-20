import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f, axs = plt.subplots(ncols=2, figsize=(9, 4), sharey=True, layout='tight')
    sh_rp = pd.read_csv('./cache/random_pairs/sh_portfolios.csv', index_col=0)
    sh_rp.sort_values('method', ascending=False, inplace=True)
    # rp = pd.read_csv('./cache/sh_portfolios.csv', index_col=0)
    sh_rp['improve'] = sh_rp['factor'] - 1
    sns.lineplot(sh_rp, x='k', y='improve', hue='method', errorbar='ci', ax=axs[0])
    # axs[0].set_yscale('log')
    axs[0].set_title('shanghai')
    la_rp = pd.read_csv('./cache/random_pairs/la_portfolios.csv', index_col=0)
    # rp = pd.read_csv('./cache/sh_portfolios.csv', index_col=0)
    la_rp['improve'] = la_rp['factor'] - 1
    sns.lineplot(la_rp, x='k', y='improve', hue='method', errorbar='ci', ax=axs[1])
    axs[1].set_yscale('log')
    axs[1].set_title('los angeles')
    plt.savefig('sh_la_ratios.png')
    

