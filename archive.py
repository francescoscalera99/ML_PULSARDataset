# ******************************************************************************************************************** #
# This file has the only purpose of storing old functions that won't probably be useful anymore, but they're here just #
# in case.                                                                                                             #
# ******************************************************************************************************************** #

plt = np = sns = itertools = colors6 = None


def create_heatmap(dataset, labels, cmap='Reds', title=None):
    """
    :param dataset:
    :param cmap:
    :param title:
    :return:
    """
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches((15, 5))
    heatmap = np.abs(np.corrcoef(dataset))
    axs[0].set_title("Whole dataset", size='xx-large')
    sns.heatmap(heatmap, cmap='Greys', annot=True, ax=axs[0])

    heatmap = np.abs(np.corrcoef(dataset[:, labels == 1]))
    axs[1].set_title("Positive class", size='xx-large')
    sns.heatmap(heatmap, cmap='Oranges', annot=True, ax=axs[1])

    heatmap = np.abs(np.corrcoef(dataset[:, labels == 0]))
    axs[2].set_title("Negative class", size='xx-large')
    sns.heatmap(heatmap, cmap='Blues', annot=True, ax=axs[2])
    fig.tight_layout()
    fig.show()
    fig.savefig(fname=f'outputs/gauss_heatmap')



def create_scatterplots(training_data, training_labels, datatype=None):
    num_features = training_data.shape[0]
    num_classes = len(set(training_labels))
    colors = ['red', 'blue']

    titles = ['1. Mean of the integrated profile',
              '2. Standard deviation of the integrated profile',
              '3. Excess kurtosis of the integrated profile',
              '4. Excess kurtosis of the integrated profile',
              '5. Mean of the DM-SNR curve',
              '6. Standard deviation of the DM-SNR curve',
              '7. Excess kurtosis of the DM-SNR curve',
              '8. Skewness of the DM-SNR curve']

    for n, (i, j) in enumerate(itertools.combinations(range(num_features), 2)):
        for k in range(num_classes):
            plt.scatter(training_data[i, training_labels == k], training_data[j, training_labels == k], color=colors[k],
                        label=f"Class {int(k)}")
        plt.xlabel(titles[i])
        plt.ylabel(titles[j])
        plt.legend()
        plt.title(f"Plot {n + 1}")
        # plt.show()
        plt.savefig(fname=f'outputs/scatter/{datatype}Figure{n + 1}')
        plt.cla()


def plot_tuningGMM():
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [None, 7]
    pis = [0.1, 0.5, 0.9]
    components_values = [rf"${2 ** i}$" for i in range(9)]

    # the datasets are
    # * raw, no PCA
    # * raw, PCA7
    # * gau, no PCA
    # * gau, PCA7

    # in each figure, given PCA and variant, we will have raw and gaussian for every value of G
    # => in each plot there are 8*2 = 16 bars
    # we will have 6 plots

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 22,
        "xtick.labelsize": 30,
        "ytick.labelsize": 20,
        "legend.fontsize": 15
    })

    n = len(raw)  # Number of bars to plot
    w = .3  # With of each column
    x = np.arange(len(components_values))  # Center position of group on x axis
    y = np.arange(0.0, 1.1, 0.2)
    print(y)

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(27, 18)

    for j, (variant, m) in enumerate(itertools.product(variants, m_values)):
        for i, r in enumerate(raw):
            DCFs = np.load(f"simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}.npy")
            position = x + (w * (1 - n) / 2) + i * w
            axs[j // 3, j % 3].bar(position, DCFs, width=w, edgecolor='black')
        axs[j // 3, j % 3].set_xticks(x, components_values)
        axs[j // 3, j % 3].set_yticks(y)

    plt.tight_layout()
    plt.show()
    # plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f'))


def bayes_error_plots(classifier):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    minDCF = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_minDCF.npy")
    actDCF = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_actDCF.npy")
    # actDCF_cal = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_actDCF_Calibrated.npy")

    plt.clf()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        # "axes.titlesize": 22,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        "legend.fontsize": 12
    })

    plt.plot(effPriorLogOdds, minDCF, label=r"$minDCF$", color="red")
    plt.plot(effPriorLogOdds, actDCF, label=r"$actDCF$", color="blue")
    # plt.plot(effPriorLogOdds, actDCF_cal, label=r"$actDCF$ (cal.)", color="blue", linestyle="dashed")
    plt.legend()
    # plt.title(classifier.__name__)
    plt.xlabel(r"$\log{\frac{\widetilde{\pi}}{(1 - \widetilde{\pi})}}$")
    plt.ylabel(r"$DCF$")
    plt.savefig(fname=f"outputs/bayes_error_plots/beforecal_{classifier.__name__}")
    plt.tight_layout()
    plt.show()


def plot_tuningGMM_evaluation():
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [None, 7]
    pis = [0.5, 0.1, 0.9]
    components_values = [2 ** i for i in range(9)]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 30,
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    fig1, axs1 = plt.subplots(3, 2)
    fig1.set_size_inches(12, 10)

    fig2, axs2 = plt.subplots(3, 2)
    fig2.set_size_inches(12, 10)

    y = np.arange(0.0, 1.1, 0.2)

    axs = [axs1, axs2]

    for i, (variant, m) in enumerate(itertools.product(variants, m_values)):
        for j, (p, r) in enumerate(itertools.product(pis, raw)):
            pp = '' if p == 0.5 else "_pi" + str(p).replace('.', '-')
            DCFs = np.load(f"../simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}{pp}.npy")
            DCFs_evaluation = np.load(f"../simulations/evaluation/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}_pi{str(p).replace('.', '-')}.npy")
            label = r"$\widetilde{\pi}=$" + f"{p}, {'raw' if r else 'gau'}"
            axs[int(r)][i // 2, i % 2].plot(components_values, DCFs, label=label, color=colors6[j], linewidth=3)
            axs[int(r)][i // 2, i % 2].plot(components_values, DCFs_evaluation, linestyle="dashed", label=label+"(eval.)", color=colors6[j], linewidth=3)
            axs[int(r)][i // 2, i % 2].set_xscale('log', base=2)
            axs[int(r)][i // 2, i % 2].set_xticks(components_values)
            axs[int(r)][i // 2, i % 2].set_yticks(y)
            if i // 2 == 2:
                axs[int(r)][i // 2, i % 2].set_xlabel("Number of components")
            if i % 2 == 0:
                axs[int(r)][i // 2, i % 2].set_ylabel(r"$DCF$")
            v = 'tied full-cov' if variant == 'tied' else variant

            pca = f"PCA ($m={m}$)" if m is not None else "no PCA"
            axs[int(r)][i // 2, i % 2].set_title(rf"{v}, {pca}", size=20)

        # axs[i // 2, i % 2].legend(loc='upper right', framealpha=0.5)

    fig1.tight_layout()
    fig1.show()
    fig1.savefig(fname="../outputs/evaluation/tuning_GMM2_gau", dpi=180)

    fig2.tight_layout()
    fig2.show()
    fig2.savefig(fname="../outputs/evaluation/tuning_GMM2_raw", dpi=180)

    label_params1 = axs1[0, 0].get_legend_handles_labels()
    figl1, axl1 = plt.subplots(figsize=(6.5, 10))
    axl1.axis(False)
    axl1.legend(*label_params1, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl1.show()
    figl1.savefig(fname="../outputs/evaluation/tuning_GMM_legend_gau")

    label_params2 = axs2[0, 0].get_legend_handles_labels()
    figl2, axl2 = plt.subplots(figsize=(6.5, 10))
    axl2.axis(False)
    axl2.legend(*label_params2, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl2.show()
    figl2.savefig(fname="../outputs/evaluation/tuning_GMM_legend_raw")


def plot_tuningGMM_evaluation2():
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [None, 7]
    pis = [0.5, 0.1, 0.9]
    components_values = [2 ** i for i in range(9)]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 30,
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    fig1, axs1 = plt.subplots(3, 2)
    fig1.set_size_inches(12, 10)

    fig2, axs2 = plt.subplots(3, 2)
    fig2.set_size_inches(12, 10)

    fig3, axs3 = plt.subplots(3, 2)
    fig3.set_size_inches(12, 10)

    y = np.arange(0.0, 1.1, 0.2)

    axs = [axs1, axs2, axs3]

    for i, (variant, m) in enumerate(itertools.product(variants, m_values)):
        for j, (p, r) in enumerate(itertools.product(pis, raw)):
            pp = '' if p == 0.5 else "_pi" + str(p).replace('.', '-')
            DCFs = np.load(f"../simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}{pp}.npy")
            DCFs_evaluation = np.load(f"../simulations/evaluation/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}_pi{str(p).replace('.', '-')}.npy")
            label = r"$\widetilde{\pi}=$" + f"{p}, {'raw' if r else 'gau'}"
            axs[pis.index(p)][i // 2, i % 2].plot(components_values, DCFs, label=label, color=colors6[j], linewidth=3)
            axs[pis.index(p)][i // 2, i % 2].plot(components_values, DCFs_evaluation, linestyle="dashed", label=label+"(eval.)", color=colors6[j], linewidth=3)
            axs[pis.index(p)][i // 2, i % 2].set_xscale('log', base=2)
            axs[pis.index(p)][i // 2, i % 2].set_xticks(components_values)
            axs[pis.index(p)][i // 2, i % 2].set_yticks(y)
            if i // 2 == 2:
                axs[pis.index(p)][i // 2, i % 2].set_xlabel("Number of components")
            if i % 2 == 0:
                axs[pis.index(p)][i // 2, i % 2].set_ylabel(r"$DCF$")
            v = 'tied full-cov' if variant == 'tied' else variant

            pca = f"PCA ($m={m}$)" if m is not None else "no PCA"
            axs[pis.index(p)][i // 2, i % 2].set_title(rf"{v}, {pca}", size=20)

        # axs[i // 2, i % 2].legend(loc='upper right', framealpha=0.5)

    fig1.tight_layout()
    fig1.show()
    fig1.savefig(fname="../outputs/evaluation/tuning_GMM2_pi0-5", dpi=180)

    fig2.tight_layout()
    fig2.show()
    fig2.savefig(fname="../outputs/evaluation/tuning_GMM2_pi0-1", dpi=180)

    fig3.tight_layout()
    fig3.show()
    fig3.savefig(fname="../outputs/evaluation/tuning_GMM2_pi0-9", dpi=180)

    label_params1 = axs1[0, 0].get_legend_handles_labels()
    figl1, axl1 = plt.subplots(figsize=(6.5, 10))
    axl1.axis(False)
    axl1.legend(*label_params1, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl1.show()
    figl1.savefig(fname="../outputs/evaluation/tuning_GMM_legend_pi0-5")

    label_params2 = axs2[0, 0].get_legend_handles_labels()
    figl2, axl2 = plt.subplots(figsize=(6.5, 10))
    axl2.axis(False)
    axl2.legend(*label_params2, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl2.show()
    figl2.savefig(fname="../outputs/evaluation/tuning_GMM_legend_pi0-1")

    label_params3 = axs3[0, 0].get_legend_handles_labels()
    figl3, axl3 = plt.subplots(figsize=(6.5, 10))
    axl3.axis(False)
    axl3.legend(*label_params3, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl3.show()
    figl3.savefig(fname="../outputs/evaluation/tuning_GMM_legend_pi0-9")
