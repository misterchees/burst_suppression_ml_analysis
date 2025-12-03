import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, beta, skewnorm


def plot_kurtosis_visuals_smooth():
    """
    Erstellt einen Plot, der drei Verteilungen mit unterschiedlicher Kurtosis
    (alle glatt, gleicher Fläche, Mittelwert=0 und Varianz=1) vergleicht.

    - Mesokurtisch: Normalverteilung (Exzess-Kurtosis = 0)
    - Leptokurtisch: Student's t-Verteilung (Exzess-Kurtosis > 0)
    - Platykurtisch: Beta-Verteilung (Exzess-Kurtosis < 0)
    """

    # 1. Datenpunkte (X-Achse) generieren
    x = np.linspace(-4.5, 4.5, 1000)

    # 2. Verteilungen definieren (alle standardisiert auf Mean=0, Var=1)

    # --- Mesokurtisch (Normalverteilung) ---
    # Mittelwert=0, Standardabweichung=1 (Varianz=1)
    y_norm = norm.pdf(x, loc=0, scale=1)

    # --- Leptokurtisch (Student's t-Verteilung) ---
    # (Identisch zum vorherigen Code)
    df_t = 5
    scale_t = np.sqrt((df_t - 2) / df_t)
    y_leptokurtic = t.pdf(x, df=df_t, loc=0, scale=scale_t)
    # Exzess-Kurtosis für t(df=5) = 6

    # --- Platykurtisch (Beta-Verteilung) ---
    # Wir verwenden Beta(a=2, b=2).
    a_beta, b_beta = 2, 2

    # Eine Standard-Beta(a,b) (auf [0,1]) hat:
    # Mittelwert: a / (a+b) = 2 / 4 = 0.5
    # Varianz: (a*b) / ((a+b)**2 * (a+b+1)) = (4) / (16 * 5) = 4/80 = 0.05

    # Wir müssen sie auf Mean=0 und Var=1 standardisieren.
    # Y = loc + scale * X

    # 1. Varianz anpassen:
    # Var(Y) = 1 = scale**2 * Var(X)
    # 1 = scale**2 * 0.05
    # scale = sqrt(1 / 0.05) = sqrt(20)
    scale_b = np.sqrt(20)

    # 2. Mittelwert anpassen:
    # E[Y] = 0 = loc + scale * E[X]
    # 0 = loc + sqrt(20) * 0.5
    # loc = -0.5 * sqrt(20)
    loc_b = -0.5 * scale_b

    y_platykurtic = beta.pdf(x, a=a_beta, b=b_beta, loc=loc_b, scale=scale_b)

    # Exzess-Kurtosis für Beta(2,2) = -0.667
    ex_kurt_beta = -2 / 3

    # 3. Plot erstellen
    plt.figure(figsize=(12, 7))

    # Plotten der Verteilungen
    plt.plot(x, y_norm, 'r-', lw=2.5, label='Mesokurtic')
    plt.plot(x, y_leptokurtic, 'b--', lw=2.5, label=f'Leptokurtic')
    plt.plot(x, y_platykurtic, 'g-.', lw=2.5, label=f'Platykurtic')

    # 4. Plot anpassen
    plt.title('Kurtosis Visualization', fontsize=16)
    # plt.xlabel('Value', fontsize=12)
    # plt.ylabel('Density', fontsize=12)

    # Y-Achse anpassen (Beta(2,2) ist etwas höher als Uniform)
    plt.ylim(0, 0.5)
    plt.xlim(x.min(), x.max())

    plt.legend(fontsize=12)

    # Entfernt die Zahlen/Werte auf der X-Achse
    plt.xticks([])

    # Entfernt die Zahlen/Werte auf der Y-Achse
    plt.yticks([])

    # Zeigt den Plot an
    plt.show()


def plot_skewness_visuals():
    """
    Erstellt einen Plot mit zwei nebeneinander liegenden Diagrammen,
    um negative und positive Schiefe (Skewness) zu visualisieren.

    - Jedes Diagramm vergleicht eine schiefe Verteilung (skewnorm)
      mit einer Normalverteilung (skew=0).
    - Alle Verteilungen sind auf Mittelwert=0 und Varianz=1 standardisiert.
    """

    # 1. Datenpunkte (X-Achse) generieren
    x = np.linspace(-4, 4, 1000)

    # 2. Basis-Verteilung (Normal, Schiefe=0)
    # Bereits standardisiert auf Mean=0, Var=1
    y_norm = norm.pdf(x, loc=0, scale=1)

    # 3. Plot-Fenster mit 2 Subplots (1 Zeile, 2 Spalten) erstellen
    # 'ax1' ist das linke Diagramm, 'ax2' das rechte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 4. Linker Plot: Negative Schiefe (Links-schief) ---

    a_neg = -5  # Starker negativer Skew-Parameter

    # Standardisierung (genau wie bei Kurtosis-Plot)
    mean_neg, var_neg = skewnorm.stats(a=a_neg, moments='mv')
    scale_neg = 1.0 / np.sqrt(var_neg)
    loc_neg = -mean_neg * scale_neg
    y_neg_skew = skewnorm.pdf(x, a=a_neg, loc=loc_neg, scale=scale_neg)

    # Den tatsächlichen Schiefe-Wert für die Legende abrufen
    skew_val_neg = skewnorm.stats(a=a_neg, loc=loc_neg, scale=scale_neg, moments='s')

    ax1.plot(x, y_neg_skew, 'r-', lw=2.5, label=f'Negativ Schief (Schiefe={skew_val_neg:.2f})')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5) # Vertikale Linie
    ax1.set_title('Negative Skewness', fontsize=14)

    # --- 5. Rechter Plot: Positive Schiefe (Rechts-schief) ---

    a_pos = 5  # Starker positiver Skew-Parameter

    # Standardisierung
    mean_pos, var_pos = skewnorm.stats(a=a_pos, moments='mv')
    scale_pos = 1.0 / np.sqrt(var_pos)
    loc_pos = -mean_pos * scale_pos
    y_pos_skew = skewnorm.pdf(x, a=a_pos, loc=loc_pos, scale=scale_pos)

    # Den tatsächlichen Schiefe-Wert für die Legende abrufen
    skew_val_pos = skewnorm.stats(a=a_pos, loc=loc_pos, scale=scale_pos, moments='s')

    ax2.plot(x, y_pos_skew, 'r-', lw=2.5, label=f'Positiv Schief (Schiefe={skew_val_pos:.2f})')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_title('Positive Skewness', fontsize=14)

    # Entfernt Zahlen auf Achsen für beide Plots
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()


# 5. Funktion aufrufen
if __name__ == "__main__":
    # plot_kurtosis_visuals_smooth()
    plot_skewness_visuals()