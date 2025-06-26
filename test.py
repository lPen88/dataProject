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

print(df.info())


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
df_plot['Survived'] = df_plot['Survived'].map({0: 'Not Survived', 1: 'Survived'})
survival_palette = {'Survived': '#7ec8e3', 'Not Survived': '#ffd8b1'}  # light blue, light orange
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
survival_stats = (
    df.dropna(subset=['Survived'])
      .groupby(['Sex', 'Pclass'])['Survived']
      .agg(['mean', 'sum', 'count'])
      .reset_index()
)
survival_stats['SurvivalRate'] = survival_stats['mean'] * 100

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=survival_stats,
    x='Pclass',
    y='SurvivalRate',
    hue='Sex'
)
plt.ylabel('Survival Rate (%)')
plt.xlabel('Passenger Class')
plt.title('Survival Rate by Sex and Passenger Class')
plt.ylim(0, 100)
plt.legend(title='Sex')

for bar, (_, group) in zip(ax.patches, survival_stats.iterrows()):
    survivors = int(group['sum'])
    total = int(group['count'])
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        f"{survivors}/{total}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.show()


# roba riguardo la famiglia

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 per includere il passeggero stesso
df=df.drop(columns=['SibSp', 'Parch'])

print(df['FamilySize'].value_counts())

#questo è diviso per la dimensione della famiglia
#perchè se hai una famiglia di 4, vuol dire che ci sono altre 3 persone con familySize=4
#quindi se li vado a contare mi troverei più famiglie da 4 persone rispetto a quelle effettive
df_plot = df.groupby(['FamilySize', 'Pclass']).size().reset_index(name='Count')
df_plot['CountPerPerson'] = df_plot['Count'] / df_plot['FamilySize']
plt.figure(figsize=(10, 6))
sns.barplot(data=df_plot, x='FamilySize', y='CountPerPerson', hue='Pclass')
plt.title('Number of Passengers per Person by Family Size and Passenger Class')
plt.ylabel('Number of Passengers per Person')
plt.xlabel('Family Size')
plt.legend(title='Passenger Class')
plt.show()

# Plot survival rate by FamilySize and Pclass
survival_by_family = (
    df.dropna(subset=['Survived'])
      .groupby(['FamilySize', 'Pclass'])['Survived']
      .mean()
      .reset_index()
)
survival_by_family['Survived'] = survival_by_family['Survived'] * 100
plt.figure(figsize=(10, 6))
sns.barplot(data=survival_by_family, x='FamilySize', y='Survived', hue='Pclass')
plt.title('Survival Rate by Family Size and Passenger Class')
plt.ylabel('Survival Rate (%)')
plt.xlabel('Family Size')
plt.legend(title='Passenger Class')
plt.ylim(0, 100)
plt.show()


# non so cosa si possa fare con il nome, ho pensato possiamo tipo estrarre il titolo e vedere se ha un impatto sulla sopravvivenza
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)

# calcolp il tasso di sopravvivenza per ogni titolo
title_survival = (
    df.dropna(subset=['Survived'])
      .groupby('Title')['Survived']
      .agg(['mean', 'sum', 'count'])
      .reset_index()
)
title_survival['Survived'] = title_survival['mean'] * 100

plt.figure(figsize=(12, 6))
order = title_survival.sort_values('Survived', ascending=False)['Title']
ax = sns.barplot(
    data=title_survival,
    x='Title',
    y='Survived',
    order=order
)
plt.ylabel('Survival Rate (%)')
plt.xlabel('Title')
plt.title('Survival Rate by Title')
plt.ylim(0, 100)
plt.xticks(rotation=45)

for i, row in title_survival.set_index('Title').loc[order].reset_index().iterrows():
    survived = int(row['sum'])
    total = int(row['count'])
    ax.text(i, row['Survived'] + 2, f"{survived}/{total}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()

# boh pare di si ma è roba che vedi già sul sesso e classe
# perchè titoli femminili/"nobili" hanno un tasso di sopravvivenza più alto



# vediamo qualcosa con il porto di imbarco ma non credo che abbia un impatto

embarked_labels = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}

embarked_survival = (
    df.dropna(subset=['Survived', 'Embarked'])
      .groupby('Embarked')['Survived']
      .agg(['mean', 'sum', 'count'])
      .reset_index()
)
embarked_survival['Survived'] = embarked_survival['mean'] * 100
embarked_survival['EmbarkedFull'] = embarked_survival['Embarked'].map(embarked_labels)

plt.figure(figsize=(8, 5))
order = ['Cherbourg', 'Queenstown', 'Southampton']
ax = sns.barplot(
    data=embarked_survival,
    x='EmbarkedFull',
    y='Survived',
    order=order,
    palette='pastel'
)
plt.ylabel('Survival Rate (%)')
plt.xlabel('Port of Embarkation')
plt.title('Survival Rate by Embarkation Port')
plt.ylim(0, 100)

for i, row in embarked_survival.set_index('EmbarkedFull').loc[order].reset_index().iterrows():
    survived = int(row['sum'])
    total = int(row['count'])
    ax.text(i, row['Survived'] + 2, f"{survived}/{total}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()

# boh non vedo come possa essere rilevante ma almeno il disegnetto l'ho fatto



# appariamo le cabine tirandoci fuori il ponte
# ogni anima su internet dice che la lettera della cabina indica il ponte
# https://www.encyclopedia-titanica.org/titanic-deckplans dice che
# i ponti A, B e C erano occupate solo da cabine di prima classe,
# D e E da cabine di tutte le classi
# F e G occupate da cabine di seconda e terza classe

# poi sul ponte più alto ("Boat Deck") ci sono le cabine T, U, W, X, Y, Z
# nel dataset compare una sola volta la cabina T di uno listato in prima classe, sopravvissuto
# lo metto in A perchè è piuttosto simile alle entry di prima classe

df['Deck'] = df['Cabin'].apply(lambda x: 'A' if pd.notna(x) and x[0] == 'T' else (x[0] if pd.notna(x) else 'X'))
df=df.drop(columns=['Cabin'])

# composizione dei ponti in base alla classe

deck_pclass_counts = df.groupby(['Deck', 'Pclass']).size().unstack(fill_value=0)
deck_totals = deck_pclass_counts.sum(axis=1)
deck_pclass_percent = deck_pclass_counts.divide(deck_totals, axis=0) * 100
pclass_colors = {1: '#d8a6a6', 2: '#c46666', 3: '#a00000'}
colors = [pclass_colors[p] for p in sorted(deck_pclass_percent.columns)]
ax = deck_pclass_percent.plot(kind='bar', stacked=True, color=colors)
plt.ylabel('Deck composition (%)')
plt.title('Percentage of Pclass per Deck')
plt.tight_layout()
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

for i, (deck, total) in enumerate(deck_totals.items()):
    ax.text(i, 102, f'{total}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.ylim(0, 110)
plt.show()

# dal disegnetto vedi che quelle con cabin mancante sono per lo più di terza classe

# tasso di sopravvivenza per ponte

survival_rate_per_deck = df.groupby('Deck')['Survived'].mean() * 100
survival_rate_per_deck = survival_rate_per_deck.sort_index()
deck_counts = df.groupby('Deck')['Survived'].agg(['sum', 'count']).loc[survival_rate_per_deck.index]

plt.figure(figsize=(8, 5))
ax = plt.gca()
for i, deck in enumerate(survival_rate_per_deck.index):
    ax.add_patch(
        plt.Rectangle((i - 0.4, 0), 0.8, 100, color='#c46666', alpha=0.15, zorder=0)
    )

bars = sns.barplot(x=survival_rate_per_deck.index, y=survival_rate_per_deck.values, color="steelblue", zorder=1, ax=ax)
plt.ylabel('Percentage of Survivors (%)')
plt.xlabel('Deck')
plt.title('Percentage of Survivors per Deck')
plt.ylim(0, 110)
plt.tight_layout()

for i, (deck, rate) in enumerate(survival_rate_per_deck.items()):
    survivors = int(deck_counts.loc[deck, 'sum'])
    total = int(deck_counts.loc[deck, 'count'])
    bar_height = rate
    ax.text(i, bar_height + 2, f"{survivors}/{total}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()


# dettaglio del ponte X visto che è quello con più entry e composto da tutte e tre le classi

deck_x = df[df['Deck'] == 'X']
survival_pclass = deck_x.groupby('Pclass')['Survived'].mean() * 100
survival_pclass = survival_pclass.sort_index()
counts = deck_x.groupby('Pclass')['Survived'].agg(['sum', 'count']).loc[survival_pclass.index]

plt.figure(figsize=(8, 5))
ax = plt.gca()

for i, pclass in enumerate(survival_pclass.index):
    ax.add_patch(
        plt.Rectangle((i - 0.4, 0), 0.8, 100, color='#c46666', alpha=0.15, zorder=0)
    )

sns.barplot(x=survival_pclass.index, y=survival_pclass.values, color="steelblue", ax=ax, zorder=1)
plt.ylabel('Percentage of Survivors (%)')
plt.xlabel('Pclass')
plt.title('Percentage of Survivors per Pclass (Deck X)')
plt.tight_layout()

for i, (pclass, row) in enumerate(counts.iterrows()):
    survivors = int(row['sum'])
    total = int(row['count'])
    ax.text(i, survival_pclass.iloc[i] + 2, f"{survivors}/{total}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.ylim(0, 110)
plt.show()


# distribuzione dei prezzi dei biglietti
plt.figure(figsize=(8, 5))
sns.histplot(df['Fare'].dropna(), bins=40, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Fare Distribution')
plt.show()


# distribuzione dei prezzi dei biglietti per classe
plt.figure(figsize=(8, 5))
for pclass in sorted(df['Pclass'].unique()):
    sns.kdeplot(
        df[df['Pclass'] == pclass]['Fare'].dropna(),
        label=f'Class {pclass}',
        linewidth=2
    )
plt.xlabel('Fare')
plt.ylabel('Density')
plt.title('Fare Distribution by Passenger Class')
plt.ylim(0, 0.15)
plt.legend(title='Pclass')
plt.show()




# fare medio per dimensione della famiglia e classe
fare_by_family = (
    df.groupby(['FamilySize', 'Pclass'])['Fare']
      .mean()
      .reset_index()
)
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=fare_by_family, x='FamilySize', y='Fare', hue='Pclass')
plt.title('Average Fare by Family Size and Passenger Class')
plt.ylabel('Average Fare')
plt.xlabel('Family Size')
plt.legend(title='Passenger Class')

plt.show()
