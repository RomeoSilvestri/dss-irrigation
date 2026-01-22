import itertools
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
import torch
import sys
sys.path.append(os.path.abspath('src'))

from dss_irrigation.anfis.sanfis import SANFIS



def train_eval_kfold(membfuncs, lr, epochs, train_idx, val_idx, seed):

    torch.manual_seed(seed)

    model = SANFIS(membfuncs=membfuncs, n_input=4, scale='Std')
    loss_function = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    X_train_fold, y_train_fold = X_tensor[train_idx], y_tensor[train_idx]
    X_valid_fold, y_valid_fold = X_tensor[val_idx], y_tensor[val_idx]

    history = model.fit(train_data=[X_train_fold, y_train_fold],
                        valid_data=[X_valid_fold, y_valid_fold],
                        optimizer=optimizer,
                        loss_function=loss_function,
                        epochs=epochs)

    return history['train_curve'], history['valid_curve']

def result_kfold(membfuncs, lr, epochs):
    train_losses_per_fold = []
    valid_losses_per_fold = []

    for train_idx, val_idx in kf.split(X_tensor):
        train_curve, valid_curve = train_eval_kfold(membfuncs, lr, epochs, train_idx, val_idx, SEED)
        train_losses_per_fold.append(train_curve)
        valid_losses_per_fold.append(valid_curve)

    mean_train_loss = np.mean(train_losses_per_fold, axis=0)
    mean_valid_loss = np.mean(valid_losses_per_fold, axis=0)

    return {
        'membfuncs': membfuncs,
        'learning_rate': lr,
        'epochs': epochs,
        'mean_train_loss': mean_train_loss,
        'mean_val_loss': mean_valid_loss
    }


df1 = pd.read_excel("data/train_data.xlsx", header=0)
df2 = pd.read_excel("data/val_data.xlsx", header=0)

df = pd.concat([df1, df2], ignore_index=True)


MEMBFUNCS_HYBRID = [
    {'function': 'hybrid', 'n_memb': 3,
     'params': {
         'low': {'c': {'value': [-0.5], 'trainable': True}, 'gamma': {'value': [-5.0], 'trainable': True}},
         'center': {'mu': {'value': [0.0], 'trainable': True}, 'sigma': {'value': [1.0], 'trainable': True}},
         'high': {'c': {'value': [0.5], 'trainable': True}, 'gamma': {'value': [5.0], 'trainable': True}}}}
] * 4

MEMBFUNCS_BELL = [
    {'function': 'bell', 'n_memb': 3,
     'params': {'c': {'value': [-1.0, 0.0, 1.0], 'trainable': True},
                'a': {'value': [0.2, 0.2, 0.2], 'trainable': True},
                'b': {'value': [2.0, 2.0, 2.0], 'trainable': True}}}
] * 4

MEMBFUNCS_GAUSSIAN = [
    {'function': 'gaussian', 'n_memb': 3,
     'params': {'mu': {'value': [-1.0, 0.0, 1.0], 'trainable': True},
                'sigma': {'value': [0.5, 0.5, 0.5], 'trainable': True}}}
] * 4

MEMBFUNCS_SIGMOID = [
    {'function': 'sigmoid', 'n_memb': 2,
     'params': {'c': {'value': [0.0, 0.0], 'trainable': True},
                'gamma': {'value': [-1.0, 1.0], 'trainable': True}}}
] * 4


SEED = 0
num_cores = os.cpu_count() - 2

membfuncs_list = [MEMBFUNCS_HYBRID, MEMBFUNCS_BELL, MEMBFUNCS_GAUSSIAN, MEMBFUNCS_SIGMOID]
learning_rates = [0.001, 0.0001, 0.00001, 0.000001]
epochs_list = [100, 200, 500, 1000]

param_grid = list(itertools.product(membfuncs_list, learning_rates, epochs_list))


numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].astype(float)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

results_list = Parallel(n_jobs=num_cores)(
    delayed(result_kfold)(m, lr, e) for m, lr, e in param_grid
)

best_result = min(results_list, key=lambda x: x['mean_val_loss'][-1])

best_membfuncs = best_result['membfuncs']
best_lr = best_result['learning_rate']
best_epochs = best_result['epochs']
best_loss = best_result['mean_val_loss'][-1]

print(f"\nBest hyperparameters found:\n"
      f"- Membership Functions: {best_membfuncs[0]['function']}\n"
      f"- Learning Rate: {best_lr}\n"
      f"- Epochs: {best_epochs}\n"
      f"- Mean Validation Loss: {best_loss:.3f}")


torch.manual_seed(SEED)

model = SANFIS(membfuncs=best_membfuncs, n_input=4, scale='Std')
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(), lr=best_lr)

history = model.fit(train_data=[X_tensor, y_tensor],
                    valid_data=[X_tensor, y_tensor],
                    optimizer=optimizer,
                    loss_function=loss_function,
                    epochs=best_epochs)

torch.save(model, "src/dss_irrigation/anfis/sanfis_dss_model.pth")