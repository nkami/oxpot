import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
from torch_geometric.utils import from_smiles
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse


# Function to generate Morgan fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(9, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.lin(x)
        return x


def train_mpn(model, smiles_list, target_list, optimizer, criterion, epochs, device):
    data_list = []
    for smiles, target in zip(smiles_list, target_list):
        data = from_smiles(smiles)
        data.y = torch.tensor([target], dtype=torch.float)
        data_list.append(data)

    # Create DataLoader
    loader = DataLoader(data_list, batch_size=32, shuffle=True)

    model.to(device)
    model.train()
    for cur_epoch in range(epochs):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'epoch {cur_epoch}, loss: {total_loss / len(loader)}')


def evaluate_mpn(model, smiles_list, target_list, batch_size=32, device=None):
    model.to(device)
    model.eval()
    data_list = []

    # Convert SMILES and targets to PyTorch Geometric data
    for smiles, target in zip(smiles_list, target_list):
        data = from_smiles(smiles)
        data.y = torch.tensor([target], dtype=torch.float)
        data_list.append(data)

    # Create DataLoader
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    preds = []
    targets = []

    # Inference
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds.append(out.cpu().numpy())
            targets.append(data.y.cpu().numpy())

    # Flatten predictions and targets
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Calculate MAE and RMSE
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))

    return mae, rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process the path to a CSV file.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)  # Assuming you have a CSV with 'SMILES' and 'eox' columns
    X = data['SMILES'].tolist()
    y = data['eox']

    X_fingerprints = np.array([smiles_to_fingerprint(smiles) for smiles in X])

    tot_samples = len(X)
    all_idx = [i for i in range(tot_samples)]
    random.shuffle(all_idx)
    train_idx = all_idx[:int(0.8 * tot_samples)]
    test_idx = all_idx[int(0.8 * tot_samples):]

    train_idx_set = set(train_idx)
    test_idx_set = set(test_idx)
    X_train = np.array([fp for i, fp in enumerate(X_fingerprints) if i in train_idx_set])
    y_train = np.array([v for i, v in enumerate(y) if i in train_idx_set])

    X_test = np.array([fp for i, fp in enumerate(X_fingerprints) if i in test_idx_set])
    y_test = np.array([v for i, v in enumerate(y) if i in test_idx_set])

    train_smiles = [s for i, s in enumerate(X) if i in train_idx_set]
    train_targets = [s for i, s in enumerate(y) if i in train_idx_set]
    test_smiles = [s for i, s in enumerate(X) if i in test_idx_set]
    test_targets = [s for i, s in enumerate(y) if i in test_idx_set]

    # Random Forest baseline
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    print(f'calculated RF: {rf_mae, rf_rmse}')

    # Support Vector Machine baseline
    svm = SVR()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_mae = mean_absolute_error(y_test, svm_pred)
    svm_rmse = np.sqrt(mean_squared_error(y_test, svm_pred))

    print(f'calculated SVR: {svm_mae, svm_rmse}')

    # Feed-Forward Neural Network baseline
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500)
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_mae = mean_absolute_error(y_test, mlp_pred)
    mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))

    print(f'calculated MLP: {mlp_mae, mlp_rmse}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(hidden_channels=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()

    train_mpn(model, train_smiles, train_targets, optimizer, criterion, epochs=1000, device=device)
    mpn_mae, mpn_rmse = evaluate_mpn(model, test_smiles, test_targets, device=device)

    # Print results
    print(f"Random Forest MAE: {rf_mae}, RMSE: {rf_rmse}")
    print(f"SVM MAE: {svm_mae}, RMSE: {svm_rmse}")
    print(f"MLP MAE: {mlp_mae}, RMSE: {mlp_rmse}")
    print(f"GCN MAE: {mpn_mae}, RMSE: {mpn_rmse}")