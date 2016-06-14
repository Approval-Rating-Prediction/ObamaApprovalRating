"""
Predict the next poll for Obama's Approval rating

"""


import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import RandomForestRegressor

distance_dict = {'Gallup': 3, 'Ipsos/Reuters': 7,
                 'YouGov/Economist':14}


def make_date(date_series):
    #creates datetime for final date spread
    release = date_series.apply(lambda x: x.split()[-1])
    months = release.apply(lambda x: x.split('/')[0])
    days = release.apply(lambda x: x.split('/')[1])
    dates = '2015' + '-' + months + '-' + days
    return dates.apply(lambda x: pd.to_datetime(x))

def expand_days(df):
    #expand df with missing days
    #check ffill
    df = df.set_index('Date')
    df = df.reindex(pd.date_range(min(df.index), max(df.index)))
    df = df.fillna(method='ffill')
    df['Date'] = df.index
    return df

def adjust_total(total):
    new = []
    for e in total:
        if len(e) > 0:
            new.append(e[0])
        else:
            new.append(0)
    return sum(new)/len(new)

def predict(df, pollster):
    #predicts next poll in df
    cat = poll_cats[pollster]
    model = RandomForestRegressor(n_estimators = 100)
    features_train = df[feats].values
    labels_train = df['Approve'].values
    model.fit(features_train, labels_train)
    
    try:
        previous = df[df.Pollster == pollster].Approve.iloc[0]
    except:
        previous = 44
    vals = df.iloc[0][features].values
    first = np.hstack((vals, cat))
    features_test = np.hstack((first, previous)) #add feats
    #entry = df['Entry'].values[0] + 1 #added Entry
    #features_test = np.hstack((features_test, entry)) #Added Entry
    pred = model.predict(features_test)[0]
    return int(pred + 0.5)

def general_prediction(df):
    #predicts next poll without knowing pollster
    #ndf = df[~df.Pollster.isin(['Gallup', 'Rasmussen'])].copy()
    ndf = df.copy()
    simple = [x for x in feats if x not in ['PollCats', 'PreviousPoll']]
    model = RandomForestRegressor(n_estimators = 100)
    features_train = ndf[simple].values
    labels_train = ndf['Approve'].values
    model.fit(features_train, labels_train)
    #f = features + ['Entry'] #added Entry
    features_test = ndf.iloc[0][features].values
    return model.predict(features_test)[0]

def predict_report(df, name):
    #give results report for the named pollster
    entries = df.Entry.values[::-1]
    entries = entries[30:]
    
    p = []
    total = []
    for cutoff in entries:
        wdf = df[df.Entry < cutoff].copy()
        pollster = df[df.Entry == cutoff].Pollster.values
        if pollster[0] == name:
            pred = predict(wdf, pollster[0])
            actual = df[df.Entry == cutoff].Approve.values
            diff = pred - actual
            p.append(abs(diff[0]))
            total.append(abs(diff[0]))
    zero = 1.0 - len([x for x in p if x > 0])/float(len(p))
    one = 1.0 - len([x for x in p if x > 1])/float(len(p))
    two = 1.0 - len([x for x in p if x > 2])/float(len(p))
    three = 1.0 - len([x for x in p if x > 3])/float(len(p))
    return zero, one, two, three

def main(pollster):
    pred = predict(df, pollster)
    z, a, b, c = predict_report(df, pollster)
    
    print pollster
    print 'Prediction: ', pred
    print ''
    print "{}: {}%".format(pred, z)
    print "{} - {}: {}%".format(pred-1, pred+1, a)
    print "{} - {}: {}%".format(pred-2, pred+2, b)
    print "{} - {}: {}%".format(pred-3, pred+3, c)
    print ''
    return (pred, [z, a, b, c])

def distance_report():
    #print how far apart entries are for regular pollsters
    for e in regulars:
        print e
        ind = df[df.Pollster == e]
        print ind.Date - ind.Date.shift(-1)
        
def next_poll_report(days):
    one = np.timedelta64(1, 'D')
    first = df.Date.max() + one
    last = first + one * (days-1)
    dates = pd.date_range(first, last)
    return dates

def ras_schedule(previous):
    five = 432000000000000
    three = 259200000000000
    prev = list(previous.astype(int))
    if prev == [five, three, five, three]:
        return 5
    elif prev == [three, five, three, five]:
        return 5
    elif prev == [five, three, five, five]:
        return 3
    elif prev == [three, five, five, three]:
        return 5
    elif prev == [five, five, three, five]:
        return 3
    else:
        return 3

def multiple_trials(pollster, trials):
    #run main() multiple times and average the result
    results = []
    for e in range(trials):
        results.append(main(pollster)[0])
    return sum(results)/float(len(results))

#Update Functions
def update_huffpo():
    import urllib
    
    testfile = urllib.URLopener()
    url = "http://elections.huffingtonpost.com/pollster/obama-job-approval.csv"
    path = '/Users/Aaron/Documents/RCP/ObamaApp/obama-job-approval.csv'
    testfile.retrieve(url, path)    
    return 'HuffPo updated'


def update_all():
    import GallupScrape
    import RasmussenScrape
    print update_huffpo()
    return 'All polls update'


### Update Polls ###

update_all()

### Rasmussen Daily ###
ras = pd.read_csv('/Users/Aaron/Documents/RCP/ObamaApp/RasDaily.csv')
ras = ras.dropna()

ras.columns = ['Date', 'Index', 'SA', 'SD', 'Approve', 'Disapprove']
ras['Approve'] = ras['Approve'].apply(lambda x: int(x[:-1]))
ras['Disapprove'] = ras['Disapprove'].apply(lambda x: int(x[:-1]))
ras['Date'] = ras['Date'].apply(lambda x: pd.to_datetime(x))
ras = ras.dropna()

ras['Pollster'] = 'Rasmussen'
ras = ras[['Date', 'Approve', 'Disapprove', 'Pollster']]

### Gallup Daily ###

gallup = pd.read_csv('/Users/Aaron/Documents/RCP/ObamaApp/GallupDaily.csv')
gallup['Date'] = gallup['Date'].apply(lambda x: pd.to_datetime(x))

### Main DataFrame ###
hpo = pd.read_csv('/Users/Aaron/Documents/RCP/ObamaApp/obama-job-approval.csv')
df = hpo[['End Date', 'Approve', 'Disapprove', 'Mode', 'Pollster']].copy()
df['Date'] = df['End Date'].apply(lambda x: pd.to_datetime(x))
#df.columns = ['End Date', 'Approve', 'Disapprove', 'Mode', 'Pollster', 'Date']
df = df[df['Date'] > pd.to_datetime('2015-7-1')] #start date


#combine ras and df

df = df[~df.Pollster.isin(['Rasmussen', 'Gallup'])]
df = ras.append(df)
df = df.append(gallup)
df = df.sort_index(ascending=False, by='Date')
df = df[df['Date'] > pd.to_datetime('2015-7-1')]

df['ApproveAVG'] = pd.rolling_mean(df['Approve'], 5)
df['ApproveAVG'] = df['ApproveAVG'].shift(-4)  

df['DisapproveAVG'] = pd.rolling_mean(df['Disapprove'], 5) #3
df['DisapproveAVG'] = df['DisapproveAVG'].shift(-4) #-2

df['SpreadAVG'] = df['ApproveAVG'] - df['DisapproveAVG']


ents = range(1, len(df)+1)
df['Entry'] = ents[::-1]

df = df.drop(['End Date', 'Mode'], axis=1)


#fix index
ind = range(len(df))
df.index = ind

### Next Poll Report ###

for e in distance_dict.keys():
    dist = distance_dict[e]
    previous = df[df.Pollster == e].Date.max()
    following = previous + np.timedelta64(dist, 'D')
    print '{}: {}'.format(e, str(following).split()[0])
    print ''

ras = df[df.Pollster == 'Rasmussen']
previous = (ras.Date - ras.Date.shift(-1)).iloc[:4]
dist = ras_schedule(previous)
previous = ras.Date.max()
following = previous + np.timedelta64(dist, 'D')
print '{}: {}'.format('Rasmussen', str(following).split()[0])
print ''


### features ###

#Most recent
print ''
print '*** Most Recent Polls *** '
print df.head(10)
print ''


#Poll Categories
poll_cats = {}
for i, e in enumerate(df['Pollster'].unique()):
    poll_cats[e] = i
df['PollCats'] = df['Pollster'].map(poll_cats)

#Last poll from given pollster
df['PreviousPoll'] = df.groupby('PollCats')['Approve'].shift(-1)

features = ['SpreadAVG', 'ApproveAVG', 'DisapproveAVG']



feats = ['SpreadAVG1', 'ApproveAVG1', 'DisapproveAVG1', 'PollCats',
         'PreviousPoll']#, 'Entry']

for f in features:
    df[f+'1'] = df[f].shift(-1)
    
df = df.dropna(subset = ['ApproveAVG1', 'PreviousPoll'])


### Prediction ###

regulars = ['Gallup', 'Rasmussen', 'Ipsos/Reuters',
            'YouGov/Economist']

#regulars = regulars[:3]

predict_values = []
print '*** Predictions ***'
for e in regulars:
    predict_values.append(main(e))

print ''
print "General Prediction"
print general_prediction(df)
