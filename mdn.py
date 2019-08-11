import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from torch.distributions.normal import Normal
from sklearn.preprocessing import RobustScaler, StandardScaler
from utils import mape, rmse


class MDN(nn.Module):
    def __init__(self, input_dim):
        super(MDN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 6*6 from image dimension
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=.2)
        x = F.dropout(F.relu(self.fc2(x)), p=.2)
        x = self.fc3(x)
        x[:, 1] = F.elu(x[:, 1]) + 1
        return x


def _make_var_cyclic(series):
    uniques = np.unique(series)
    diff = np.arange(0, 2*np.pi, 2*np.pi/len(uniques))
    cos, sin = np.cos(diff), np.sin(diff)

    x_dict = {k: v for (k, v) in zip(uniques, cos)}
    y_dict = {k: v for (k, v) in zip(uniques, sin)}

    if isinstance(series, pd.Series):
        return series.apply(lambda x: x_dict[x]), series.apply(lambda x: y_dict[x])

    return np.array([x_dict[i] for i in series]), np.array([y_dict[i] for i in series])


def parse_date(series):
    df = pd.DataFrame()
    df['year'] = series.apply(lambda x: x.year)
    df['month_x'], df['month_y'] = _make_var_cyclic(series.apply(lambda x: x.month))
    df['weekday_x'], df['weekday_y'] = _make_var_cyclic(series.apply(lambda x: x.dayofweek))
    df['day_x'], df['day_y'] = _make_var_cyclic(series.apply(lambda x: x.day))
    df['hour_x'], df['hour_y'] = _make_var_cyclic(series.apply(lambda x: x.hour))
    return df


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


x_scaler = StandardScaler()
y_scaler = RobustScaler(quantile_range=(.1, .9))

df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv.gz', compression='gzip', parse_dates=['date_time'])
df = pd.concat((df, parse_date(df['date_time'])), axis=1)

x_train = x_scaler.fit_transform(df[['year', 'month_x', 'month_y', 'weekday_x', 'weekday_y', 'day_x', 'day_y', 'hour_x', 'hour_y']])
y_train = y_scaler.fit_transform(df[['traffic_volume']])
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=.1, shuffle=False)

x_train = torch.tensor(x_train).float().to(device=device)
x_test = torch.tensor(x_test).float().to(device=device)
y_train = torch.tensor(y_train).float().to(device=device)
y_train = y_train.view((-1, 1))

net = MDN(x_train.shape[1]).to(device=device)
net.zero_grad()


learning_rate = 1e-3
epochs = 5000
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

tempo = time.time()
for i in range(1, epochs+1):
    y_train_pred = net(x_train).float()

    loss = Normal(y_train_pred[:, 0], y_train_pred[:, 1])
    loss = -loss.log_prob(y_train.squeeze()).mean()
    print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('TRAINING TIME: {}s'.format(time.time()-tempo))

y_train = y_scaler.inverse_transform(y_train.cpu().numpy())
y_train_pred = net(x_train).cpu().detach().numpy()
y_train_pred_min = y_scaler.inverse_transform(y_train_pred[:, :1] - 2 * y_train_pred[:, 1:])
y_train_pred_max = y_scaler.inverse_transform(y_train_pred[:, :1] + 2 * y_train_pred[:, 1:])
y_train_pred = y_scaler.inverse_transform(y_train_pred[:, :1])

y_test = y_scaler.inverse_transform(y_test)
y_test_pred = net(x_test).cpu().detach().numpy()
y_test_pred_min = y_scaler.inverse_transform(y_test_pred[:, :1] - 2 * y_test_pred[:, 1:])
y_test_pred_max = y_scaler.inverse_transform(y_test_pred[:, :1] + 2 * y_test_pred[:, 1:])
y_test_pred = y_scaler.inverse_transform(y_test_pred[:, :1])

plt.plot(y_train)
plt.plot(y_train_pred)
plt.fill_between(np.arange(y_train.shape[0]),
                 y_train_pred_min.squeeze(),
                 y_train_pred_max.squeeze(),
                 color='b', alpha=.1)

plt.plot(np.arange(y_train.shape[0], df.shape[0]), y_test)
plt.plot(np.arange(y_train.shape[0], df.shape[0]), y_test_pred)
plt.fill_between(np.arange(y_train.shape[0], df.shape[0]),
                 y_test_pred_min.squeeze(),
                 y_test_pred_max.squeeze(),
                 color='b', alpha=.1)
plt.show()

print('TEST RMSE: {}'.format(rmse(y_test, y_test_pred[:, 0])))
print('TEST MAPE: {}'.format(mape(y_test, y_test_pred[:, 0])))
