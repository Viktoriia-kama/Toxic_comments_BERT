# -*- coding: utf-8 -*-
"""EDA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/Viktoriia-kama/Toxic_comments_BERT/blob/main/EDA.ipynb

# Toxic Comment Classification

## Packages Loading
"""

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
import zipfile

from wordcloud import WordCloud

import nltk
nltk.download('wordnet')


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string

"""## Data Preprocessing and Exploratory Data Analyasis

<a id='Data-loading'></a>
### 📥 Data Loading
"""

train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
test_y = pd.read_csv("/content/test_labels.csv")

"""### Data Analysis"""

train.head()

train.describe()

train['toxic'].value_counts(normalize = True)

"""Наш датасет має значний дисбаланс класів. Токсичні коментарі складають менше 10% від усіх коментарів, тоді як нетоксичні коментарі становлять понад 90%. Це може ускладнити навчання моделі, оскільки модель може мати тенденцію до передбачення більшості класу (нетоксичний) без правильного навчання на менш чисельному класі (токсичний).

Розгляньмо візуалізацію розподілу токсичних і нетоксичних коментарів, щоб краще зрозуміти дані. Також проаналізуємо, чи є якісь специфічні патерни або теми в токсичних коментарях, які можна використовувати для покращення моделі
"""

plt.figure(figsize=(8, 6))
sns.countplot(x='toxic', data=train, palette='viridis')
plt.title('Розподіл токсичних і нетоксичних коментарів')
plt.xlabel('Токсичність')
plt.ylabel('Кількість коментарів')
plt.xticks(ticks=[0, 1], labels=['Нетоксичний', 'Токсичний'])
plt.show()

test.head()

test_y.head()

"""Значення -1 вказує на те, що мітка для цього коментаря або не була надана, або вона є недійсною. Це може бути сигналом, що ці коментарі або не були розмічені, або розмітка відсутня."""

test_y.describe()

test_y[test_y['toxic']==-1]

"""Важливо визначити, чому в тестовому наборі є значення -1. Це можуть бути неповні або пошкоджені дані. Було вирішено, як з ними справитися — ------------------

"""

test_y.toxic.unique()

"""Для візуалізації було створено графіки, які порівнюють кількість рядків з значеннями -1, 0, і 1 у тестовому наборі даних, щоб краще зрозуміти їх розподіл і потенційний вплив на нашу модель."""

# Візуалізація кругової діаграми для ознаки 'toxic'
toxic_counts = test_y['toxic'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(toxic_counts, labels=['-1 (відсутній)', '0 (нетоксичний)', '1 (токсичний)'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF', '#99FF99'])
plt.title('Розподіл значень для ознаки "toxic" в тестовому наборі')
plt.show()

"""1. Високий рівень відсутніх даних:

58% значень -1 у нашому тестовому наборі означає, що понад половина даних має невизначені або відсутні мітки. Це може бути через помилки в розмітці або неповну інформацію. Цей високий відсоток може суттєво вплинути на результати моделі, оскільки ми не зможемо використовувати ці рядки для перевірки моделі.
2. Низька частка токсичних коментарів:

Лише 4% коментарів є токсичними (значення 1). Це показує, що тестовий набір містить дуже невелику кількість токсичних коментарів. Це може бути проблемою, якщо модель недостатньо навчена на токсичних коментарях або якщо в тестовому наборі недостатньо представлено токсичні приклади.
3. Більшість коментарів є нетоксичними:

37% коментарів є нетоксичними (значення 0). Це означає, що значна частина тестових даних містить коментарі, які вважаються нетоксичними. Це також вказує на дисбаланс між класами.
"""

train.shape

test.shape

"""1. Різниця в кількості рядків:

Тестовий набір менший за навчальний (159,571 проти 153,164). Це нормальна ситуація, але ми повинні переконатися, що тестовий набір не має пропущених значень у важливих стовпцях і що дані для тестування та навчання сумісні.
2. Відсутність міток у тестовому наборі:

Тестовий набір містить лише id і comment_text, але не містить міток токсичності. Це вказує на те, що тестовий набір використовується для перевірки моделі після її навчання на train, і ми не маємо фактичних міток для оцінки результатів.
"""

# Перевірка пропущених значень у тестовому наборі
print(test.isnull().sum())
print("_____________________________________________")

# Перевірка імен стовпців
print("Навчальний набір:", train.columns)
print("Тестовий набір:", test.columns)

# Перевірка типів даних
print("Типи даних навчального набору:\n", train.dtypes)
print("Типи даних тестового набору:\n", test.dtypes)
print("_____________________________________________")

# Перевірка унікальних значень у стовпці 'id' для тестового набору
print("Унікальні значення в 'id':", test['id'].unique())

# Перевірка першого рядка текстових коментарів
print(test['comment_text'].head())
print("_____________________________________________")

# Перевірка наявності всіх стовпців
required_columns = ['id', 'comment_text']
missing_columns = [col for col in required_columns if col not in test.columns]
if missing_columns:
    print(f"В тестовому наборі відсутні стовпці: {missing_columns}")
else:
    print("Всі необхідні стовпці присутні.")
print("_____________________________________________")

# Перевірка кількості символів в перших кількох коментарях
print(test['comment_text'].apply(len).describe())

sns.set(color_codes=True)
comment_len = train.comment_text.str.len()
sns.histplot(comment_len, kde=False, bins=20, color="steelblue")

"""Частота позначених коментарів. 'toxic' найчастіші.


"""

train_labels = train[['toxic', 'severe_toxic',
                      'obscene', 'threat', 'insult', 'identity_hate']]
label_count = train_labels.sum()

label_count.plot(kind='bar', title='Labels Frequency', color='steelblue')

"""Деякі коментарі належать одночасно декільком класам.
Це вплине на спосіб визначення класу
"""

train[train["toxic"]==1]

"""Візуалізація частоти"""

barWidth = 0.25

bars1 = [sum(train['toxic'] == 1), sum(train['obscene'] == 1), sum(train['insult'] == 1), sum(train['severe_toxic'] == 1),
         sum(train['identity_hate'] == 1), sum(train['threat'] == 1)]
bars2 = [sum(train['toxic'] == 0), sum(train['obscene'] == 0), sum(train['insult'] == 0), sum(train['severe_toxic'] == 0),
         sum(train['identity_hate'] == 0), sum(train['threat'] == 0)]

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars1, color='steelblue', width=barWidth, label='labeled = 1')
plt.bar(r2, bars2, color='lightsteelblue', width=barWidth, label='labeled = 0')

plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Toxic', 'Obscene', 'Insult', 'Severe Toxic', 'Identity Hate',
                                                       'Threat'])
plt.legend()
plt.show()

"""Загальні висновки:
* Тестовий набір виглядає правильно підготовленим і сумісним з навчальним набором, крім відсутності міток токсичності, що є звичним для тестового набору.

* Обробка тексту: Зважаючи на варіації в довжині коментарів, може бути доцільно провести попередню обробку тексту, щоб вирішити питання з довгими коментарями.

Приклад чистого коментаря
"""

print(train.comment_text[0])

"""Токсичний коментар, приклад"""

print(train[train.toxic == 1].iloc[1, 1])

rowsums = train.iloc[:, 2:].sum(axis=1)
temp = train.iloc[:, 2:-1]
train_corr = temp[rowsums > 0]
corr = train_corr.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap="Blues")

"""Маємо високу кореляцію між threat та insult та obscene.

Побудова графіку у вигляді хмари слів.

Цей підхід дозволить вам візуалізувати найпопулярніші слова в коментарях і може допомогти у розумінні найбільш вживаних термінів у вашому наборі даних.
"""

def W_Cloud(token):
    """
    Visualize the most common words contributing to the token.
    """
    threat_context = train[train[token] == 1]
    threat_text = threat_context.comment_text
    neg_text = pd.Series(threat_text).str.cat(sep=' ')
    wordcloud = WordCloud(width=1600, height=800,
                          max_font_size=200).generate(neg_text)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud.recolor(colormap="Blues"), interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Most common words assosiated with {token} comment", size=20)
    plt.show()

train_labels_words = ['toxic', 'severe_toxic',
                      'obscene', 'threat', 'insult', 'identity_hate']
for token in train_labels_words:
  W_Cloud(token.lower())

"""## Feature-engineering"""

test_labels = ["toxic", "severe_toxic", "obscene",
               "threat", "insult", "identity_hate"]

def tokenize(text):
    '''
    Tokenize text and return a non-unique list of tokenized words found in the text.
    Normalize to lowercase, strip punctuation, remove stop words, filter non-ascii characters.
    Lemmatize the words and lastly drop words of length < 3.
    '''
    text = text.lower()
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    words = nopunct.split(' ')
    # remove any non ascii
    words = [word.encode('ascii', 'ignore').decode('ascii') for word in words]
    lmtzr = WordNetLemmatizer()
    words = [lmtzr.lemmatize(w) for w in words]
    words = [w for w in words if len(w) > 2]
    return words

vector = TfidfVectorizer(ngram_range=(1, 1), analyzer='word',
                         tokenizer=tokenize, stop_words='english',
                         strip_accents='unicode', use_idf=1, min_df=10)
X_train = vector.fit_transform(train['comment_text'])
X_test = vector.transform(test['comment_text'])

vector.get_feature_names_out()[0:20]

df_test = pd.merge(test, test_y, on="id")

df_test.head()

df_test.shape

test.head()

"""Цей процес створив векторизовані дані, готові до подальшого навчання моделі"""