import pandas as pd

class Fill():
    def fill_state(self, df):
        region_state_dict = df[['region', 'state']].dropna().drop_duplicates().to_dict(orient='records')
        dic = {
        'northwest KS': 'ks',
        'ashtabula': 'oh',
        'southern WV': 'wv',
        'hanford-corcoran': 'ca',
        'glens falls': 'ny'
        }

        for d in region_state_dict:
            df.loc[df['region']==d['region'], 'state'] = d['state']

        for k, v in dic.items():
            df.loc[df['region']==k, 'state'] = v
    
        return df

    def fill_fuel(self, df):
            # filled_df = df.copy()
            df.loc[(df["type"] == "bus"),"fuel"] = df.loc[(df["type"] == "bus"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "coupe"),"fuel"] = df.loc[(df["type"] == "coupe"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "hatchback"),"fuel"] = df.loc[(df["type"] == "hatchback"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "mini-van"),"fuel"] = df.loc[(df["type"] == "mini-van"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "sedan"),"fuel"] = df.loc[(df["type"] == "sedan"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "van"),"fuel"] = df.loc[(df["type"] == "van"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "other"),"fuel"] = df.loc[(df["type"] == "other"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "offroad"),"fuel"] = df.loc[(df["type"] == "offroad"),"fuel"].fillna('gas')
            df.loc[(df['type'] == 'wagon') | (df['type'] == 'convertible'), 'fuel'] = df.loc[(df['type'] == 'wagon') | (df['type'] == 'convertible'), 'fuel'].fillna('gas')

            df.loc[(((df["type"] == "SUV")&(df["manufacturer"]== "ram"))|((df["type"] == "SUV")&(df["manufacturer"]== "ford"))),"fuel"] = df.loc[(((df["type"] == "SUV")&(df["manufacturer"]== "ram"))|((df["type"] == "SUV")&(df["manufacturer"]== "ford"))),"fuel"].fillna('diesel')
            df.loc[(((df["type"] == "truck")&(df["manufacturer"]== "ram"))|((df["type"] == "truck")&(df["manufacturer"]== "ford"))),"fuel"] = df.loc[(((df["type"] == "truck")&(df["manufacturer"]== "ram"))|((df["type"] == "truck")&(df["manufacturer"]== "ford"))),"fuel"].fillna('diesel')
            df.loc[(((df["type"] == "pickup")&(df["manufacturer"]== "ram"))|((df["type"] == "pickup")&(df["manufacturer"]== "ford"))),"fuel"] = df.loc[(((df["type"] == "pickup")&(df["manufacturer"]== "ram"))|((df["type"] == "pickup")&(df["manufacturer"]== "ford"))),"fuel"].fillna('diesel')
            
            df.loc[(df["type"] == "SUV"),"fuel"] = df.loc[(df["type"] == "SUV"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "truck"),"fuel"] = df.loc[(df["type"] == "truck"),"fuel"].fillna('gas')
            df.loc[(df["type"] == "pickup"),"fuel"] = df.loc[(df["type"] == "pickup"),"fuel"].fillna('gas')
            return df