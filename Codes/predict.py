import pandas as pd

data_dir = '../data/WithoutReviews/'
df = pd.read_hdf(data_dir + 'No_normalized_data.h5')
# df = df.groupby('deal_id').nunique()
# print(df.head())
# print((df.groupby('asin').nunique()))
# print(df[df['claim'] > 50].count(), len(df))
deal = {'1'}
cnt = 0
cuttoff = input("What is the cuttoff claim?")
for i in range(len(df)):
	d_id = df.iloc[i]['deal_id']
	if(d_id in deal):
		continue
	d_claim = df.iloc[i]['claim']
	if(d_claim > int(cuttoff)):
		cnt = cnt+1
		deal.add(d_id)
print(cnt)