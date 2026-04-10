import pandas as pd



df = pd.read_csv("raw_dataset.csv", usecols=['instruction', 'intent'])

 
df = df.rename(columns={'instruction': 'text', 'intent': 'label'})


istenen_niyetler = [
    'order_food_online', 
    'file_complaint', 
    'make_reservation', 
    'check_menu', 
    'locations',
    'leave_review'
]


df_final = df[df['label'].isin(istenen_niyetler)]
df_final = df_final.dropna()


df_final.to_csv("smash_society_dataset.csv", index=False)

print(f"Kusursuz temizlik! Toplam veri sayısı: {len(df_final)}")
print(df_final.head())