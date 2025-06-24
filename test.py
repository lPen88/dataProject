import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# li unisco così vediamo risolviamo i valori mancanti in entrambi
# poi anche perchè abbiamo più dati per vedere eventuali correlazioni
df = pd.concat([train_df, test_df], ignore_index=True)

df = df.drop(columns=['PassengerId'])


#distribuzione età (collettivo)
plt.figure(figsize=(8, 5))
df['Age'].hist(bins=30, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

#distribuzione età (per sesso)
plt.figure(figsize=(8, 5))
for sex in df['Sex'].unique():
    df[df['Sex'] == sex]['Age'].hist(bins=30, alpha=0.6, label=sex, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution by Sex')
plt.legend()
plt.show()

#distribuzione età (per classe)
plt.figure(figsize=(8, 5))
for pclass in sorted(df['Pclass'].unique()):
    df[df['Pclass'] == pclass]['Age'].hist(bins=30, alpha=0.6, label=f'Class {pclass}', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution by Passenger Class')
plt.legend()
plt.show()

#distribuzione età (per sesso e classe)
g = sns.FacetGrid(df, row="Sex", col="Pclass", margin_titles=True, height=3, aspect=1.2)
g.map(sns.histplot, "Age", bins=30, color="steelblue", alpha=0.7, kde=False)
# per avere più tacche sull'asse x
for ax in g.axes.flatten():
    ax.set_xticks(np.arange(0, 85, 10))  # Adjust range and step as needed
plt.subplots_adjust(top=0.9)
g.figure.suptitle("Distribuzione dell'età per sesso e classe (tutti i passeggeri)")
plt.show()


# stesso di sopra ma con sopravvivenza
df_plot = df.dropna(subset=['Survived', 'Age'])
# Convert 'Survived' to string for better plot labels
df_plot['Survived'] = df_plot['Survived'].map({0: 'Not Survived', 1: 'Survived'})
# Define custom colors for Survived/Not Survived
survival_palette = {'Survived': '#7ec8e3', 'Not Survived': '#ffd8b1'}  # light blue, light orange
# Plot: Age distribution by Survived, Sex, and Pclass
g = sns.FacetGrid(
    df_plot,
    row="Sex",
    col="Pclass",
    hue="Survived",
    margin_titles=True,
    height=3,
    aspect=1.2,
    palette=survival_palette
)
g.map(sns.histplot, "Age", bins=30, alpha=0.7, kde=False)
g.add_legend(title="Survival Status")
plt.subplots_adjust(top=0.9)
g.figure.suptitle("Distribuzione dell'età per sesso, classe e sopravvivenza")
plt.show()


# Calcola la percentuale di sopravvissuti per gruppo (Sex, Pclass)
survival_rates = (
    df.dropna(subset=['Survived'])
      .groupby(['Sex', 'Pclass'])['Survived']
      .mean()
      .reset_index()
)
survival_rates['Survived'] = survival_rates['Survived'] * 100

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(
    data=survival_rates,
    x='Pclass',
    y='Survived',
    hue='Sex'
)
plt.ylabel('Survival Rate (%)')
plt.title('Survival Rate by Sex and Passenger Class')
plt.ylim(0, 100)
plt.legend(title='Sex')
plt.show()