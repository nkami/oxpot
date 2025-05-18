import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
# from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, GNNExplainer
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch.nn import Linear
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import sys

from utils import from_smiles, convert_common_attributes_to_dict


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
        self.conv1 = GCNConv(226, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GIN, self).__init__()

        mlp1 = torch.nn.Sequential(
            Linear(226, hidden_channels),
            BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )

        mlp2 = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )

        mlp3 = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )

        mlp4 = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )

        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.conv3 = GINConv(mlp3)
        self.conv4 = GINConv(mlp4)

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x


def train_mpn(model, train_smiles, train_targets, valid_smiles, valid_targets, optimizer, criterion, max_epochs, device,
              patience=20):
    # Prepare training data
    data_list = []
    for smiles, target in zip(train_smiles, train_targets):
        data = from_smiles(smiles)
        data.y = torch.tensor([target], dtype=torch.float)
        data_list.append(data)
    train_loader = DataLoader(data_list, batch_size=128, shuffle=True)

    # Prepare validation data
    valid_data = []
    for smiles, target in zip(valid_smiles, valid_targets):
        data = from_smiles(smiles)
        data.y = torch.tensor([target], dtype=torch.float)
        valid_data.append(data)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)

    model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for cur_epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch
            out = model(x, edge_index, batch=batch)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                x, edge_index, batch = data.x.float(), data.edge_index, data.batch
                out = model(x, edge_index, batch=batch)
                loss = criterion(out, data.y.view(-1, 1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {cur_epoch}: train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


def evaluate_mpn(model, smiles_list, target_list, batch_size=32, device=None, output_path=None, exp_num=0):
    model.to(device)
    model.eval()

    data_list = []
    for smiles, target in zip(smiles_list, target_list):
        data = from_smiles(smiles)
        data.smiles = smiles
        data.y = torch.tensor([target], dtype=torch.float)
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    preds = []
    targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch
            out = model(x, edge_index, batch=batch)
            preds.append(out.cpu().numpy())
            targets.append(data.y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # MAE & RMSE
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))

    if output_path is not None:
        feature_explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100, lr=0.01),
            explanation_type='model',
            node_mask_type='common_attributes',
            edge_mask_type='object',
            model_config=dict(
                mode="regression",
                task_level="graph",
                return_type="raw",
            ),
        )

        all_features = {}
        for i, data in enumerate(data_list):
            current_feature_explanation = feature_explainer(x=data.x.to(device), edge_index=data.edge_index.to(device))
            current_common_attributes = current_feature_explanation.node_mask
            attributes_dict = convert_common_attributes_to_dict(current_common_attributes)
            for attributes_key, attribute_val in attributes_dict.items():
                if attributes_key in all_features:
                    all_features[attributes_key].extend(attribute_val)
                else:
                    all_features[attributes_key] = attribute_val

        # Convert to long-form DataFrame for seaborn
        records = []
        for feature, values in all_features.items():
            for val in values:
                records.append({'Feature': feature, 'Importance': val})

        df = pd.DataFrame(records)

        # Sort by median importance
        median_importance = df.groupby('Feature')['Importance'].median()
        sorted_features = median_importance.sort_values(ascending=False).index.tolist()
        df['Feature'] = pd.Categorical(df['Feature'], categories=sorted_features, ordered=True)

        # Plot
        plt.figure(figsize=(max(10, len(df['Feature'].unique()) * 0.5), 6))
        sns.violinplot(data=df, x='Feature', y='Importance', palette='pastel')

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(f"./{output_path}/violin_feature_importance_{exp_num}.png", dpi=300)


    return mae, rmse


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    output_dir = f'./results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="Process the path to a CSV file.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)  # Assuming you have a CSV with 'SMILES' and 'eox' columns
    X = data['SMILES'].tolist()
    y = data['eox']

    X_fingerprints = np.array([smiles_to_fingerprint(smiles) for smiles in X])
    print('Generated fingerprints')

    tot_samples = len(X)
    all_idx = [i for i in range(tot_samples)]

    models_results = {'rf': [], 'svm': [], 'linear_svm': [], 'mlp': [], 'gcn': [], 'gin': []}
    for i in range(5):
        random.seed(i)
        random.shuffle(all_idx)

        train_end = int(0.8 * tot_samples)
        val_end = int(0.9 * tot_samples)

        train_idx = all_idx[:train_end]
        val_idx = all_idx[train_end:val_end]
        test_idx = all_idx[val_end:]

        train_idx_set = set(train_idx)
        val_idx_set = set(val_idx)
        test_idx_set = set(test_idx)

        X_train = np.array([fp for i, fp in enumerate(X_fingerprints) if i in train_idx_set])
        y_train = np.array([v for i, v in enumerate(y) if i in train_idx_set])

        X_val = np.array([fp for i, fp in enumerate(X_fingerprints) if i in val_idx_set])
        y_val = np.array([v for i, v in enumerate(y) if i in val_idx_set])

        X_test = np.array([fp for i, fp in enumerate(X_fingerprints) if i in test_idx_set])
        y_test = np.array([v for i, v in enumerate(y) if i in test_idx_set])

        train_smiles = [s for i, s in enumerate(X) if i in train_idx_set]
        train_targets = [s for i, s in enumerate(y) if i in train_idx_set]
        valid_smiles = [s for i, s in enumerate(X) if i in train_idx_set]
        valid_targets = [s for i, s in enumerate(y) if i in train_idx_set]
        test_smiles = [s for i, s in enumerate(X) if i in test_idx_set]
        test_targets = [s for i, s in enumerate(y) if i in test_idx_set]

        # ---- Random Forest ----
        rf_params = [
            {'n_estimators': 100, 'max_depth': None, 'n_jobs': -1},
            {'n_estimators': 200, 'max_depth': 5, 'n_jobs': -1},
            {'n_estimators': 300, 'max_depth': 10, 'n_jobs': -1},
        ]
        best_rf = None
        best_val_mae = float('inf')

        for params in rf_params:
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_rf = model

        best_rf.fit(X_train, y_train)
        test_pred = best_rf.predict(X_test)
        rf_mae = mean_absolute_error(y_test, test_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        models_results['rf'].append((rf_mae, rf_rmse))
        print(f"Best RF: MAE={rf_mae:.4f}, RMSE={rf_rmse:.4f}")

        # ---- SVM ----
        svm_params = [
            {'C': 1.0, 'epsilon': 0.1, 'max_iter': 3000},
        ]
        best_svm = None
        best_val_mae = float('inf')

        for params in svm_params:
            model = SVR(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_svm = model

        best_svm.fit(X_train, y_train)
        test_pred = best_svm.predict(X_test)
        svm_mae = mean_absolute_error(y_test, test_pred)
        svm_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        models_results['svm'].append((svm_mae, svm_rmse))
        print(f"Best SVR: MAE={svm_mae:.4f}, RMSE={svm_rmse:.4f}")

        # ---- Linear SVM ----
        svm_params = [
            {'C': 1.0, 'epsilon': 0.1, 'max_iter': 3000},
            {'C': 10.0, 'epsilon': 0.2, 'max_iter': 3000},
            {'C': 100.0, 'epsilon': 0.01, 'max_iter': 3000},
        ]
        best_linear_svm = None
        best_val_mae = float('inf')

        for params in svm_params:
            model = LinearSVR(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_linear_svm = model

        best_linear_svm.fit(X_train, y_train)
        test_pred = best_linear_svm.predict(X_test)
        linear_svm_mae = mean_absolute_error(y_test, test_pred)
        linear_svm_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        models_results['linear_svm'].append((linear_svm_mae, linear_svm_rmse))
        print(f"Best Linear SVR: MAE={linear_svm_mae:.4f}, RMSE={linear_svm_rmse:.4f}")

        # ---- MLP ----
        mlp_params = [
            {'hidden_layer_sizes': (64, 64), 'max_iter': 300, 'early_stopping': True},
            {'hidden_layer_sizes': (128, 64), 'max_iter': 300, 'early_stopping': True},
            {'hidden_layer_sizes': (128, 128), 'max_iter': 300, 'early_stopping': True},
        ]
        best_mlp = None
        best_val_mae = float('inf')

        for params in mlp_params:
            model = MLPRegressor(**params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_mlp = model

        best_mlp.fit(X_train, y_train)
        test_pred = best_mlp.predict(X_test)
        mlp_mae = mean_absolute_error(y_test, test_pred)
        mlp_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        models_results['mlp'].append((mlp_mae, mlp_rmse))
        print(f"Best MLP: MAE={mlp_mae:.4f}, RMSE={mlp_rmse:.4f}")

        # ---- GCN ----
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(hidden_channels=128)
        print(f'GCN using device {device}')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = torch.nn.MSELoss()

        train_mpn(model, train_smiles, train_targets, valid_smiles, valid_targets, optimizer, criterion, max_epochs=500, device=device)
        gcn_mae, gcn_rmse = evaluate_mpn(model, test_smiles, test_targets, device=device, output_path=None, exp_num=i)
        models_results['gcn'].append((gcn_mae, gcn_rmse))
        print(f"Best GCN: MAE={gcn_mae:.4f}, RMSE={gcn_rmse:.4f}")

        # ---- GIN ----
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GIN(hidden_channels=128)
        print(f'GIN using device {device}')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = torch.nn.MSELoss()

        train_mpn(model, train_smiles, train_targets, valid_smiles, valid_targets, optimizer, criterion, max_epochs=500, device=device)
        gin_mae, gin_rmse = evaluate_mpn(model, test_smiles, test_targets, device=device, output_path=output_dir, exp_num=i)
        models_results['gin'].append((gin_mae, gin_rmse))
        print(f"Best GIN: MAE={gin_mae:.4f}, RMSE={gin_rmse:.4f}")


    # Step 1: Convert to MultiIndex DataFrame
    rows = []
    num_experiments = len(next(iter(models_results.values())))

    for i in range(num_experiments):
        row = {}
        for name, values in models_results.items():
            mae, rmse = values[i]
            row[f"{name}_mae"] = mae
            row[f"{name}_rmse"] = rmse
        rows.append(row)

    df = pd.DataFrame(rows)

    # Step 2: Add mean and std rows
    mean_row = {col: df[col].mean() for col in df.columns}
    std_row = {col: df[col].std() for col in df.columns}

    mean_row = pd.Series(mean_row, name='mean')
    std_row = pd.Series(std_row, name='std')

    # Step 3: Append summary rows
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])])

    # Step 4: Save to CSV
    df.to_csv(f"{output_dir}/results_summary_{timestamp}.csv", index_label="experiment")

    print(f'************ FINAL RESULTS ************')
    print(df)