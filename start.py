import pandas as pd

train_path = './train.txt'
df = pd.DataFrame(columns=['label', 'review'])

with open(train_path, 'r') as f:
    for line in f:
        divide = line.strip().split('\t') # divide LABEL from REVIEW
        label, review = divide
        df = df._append({'label': label, 'review': review}, ignore_index=True)

print(df.head())