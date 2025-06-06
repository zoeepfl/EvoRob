import pickle
import matplotlib.pyplot as plt  # ✅ Nécessaire pour plt.show()

with open('reward_data/es_best_and_mean_plot_random.pkl', 'rb') as f:
    fig = pickle.load(f)

plt.figure(fig.number)  # Optionnel : réactive la figure dans matplotlib
plt.show()  # ✅ Bloque jusqu'à ce que la figure soit fermée
