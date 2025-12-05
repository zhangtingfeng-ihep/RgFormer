import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pandas as pd
import logging
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(filename='C:\\SASBDB\\model\\34\\train.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model class without channel attention, positional encoding, and residual block
class ImprovedSAXSModel(nn.Module):
    def __init__(self, input_dim=2, d_model=512, nhead=16, num_layers=4, dim_feedforward=2048, max_length=2617,
                 dropout=0.2):
        super(ImprovedSAXSModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(2, 2)
        self.embedding = nn.Linear(64 * 3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        conv1_out = torch.relu(self.conv1(x))
        conv2_out = torch.relu(self.conv2(x))
        conv3_out = torch.relu(self.conv3(x))
        x = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        x = self.pool(x)
        x = x.transpose(1, 2)
        seq_len_after_pool = x.size(1)
        x = self.embedding(x)

        mask = mask[:, ::2]
        if mask.size(1) > seq_len_after_pool:
            mask = mask[:, :seq_len_after_pool]
        elif mask.size(1) < seq_len_after_pool:
            padding = torch.zeros(mask.size(0), seq_len_after_pool - mask.size(1)).to(mask.device)
            mask = torch.cat([mask, padding], dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=mask == 0)
        x = self.ln(x.mean(dim=1))
        x = self.mlp(x)
        return x

# Define the dataset class with improved weights
class SAXSDataset(Dataset):
    def __init__(self, curve_data, rg_dict, max_length):
        self.curve_data = curve_data
        self.rg_dict = rg_dict  # Directly use Rg as target
        self.max_length = max_length
        self.codes = list(curve_data.keys())

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        data = self.curve_data[code]
        rg = self.rg_dict[code]
        padded_data = np.zeros((self.max_length, 2), dtype=np.float32)
        seq_len = min(len(data), self.max_length)
        padded_data[:seq_len, :] = data[:seq_len, :]
        mask = np.zeros(self.max_length, dtype=np.float32)
        mask[:seq_len] = 1
        # Dynamic weight based on Rg
        weight = 1.0 / (rg + 1e-10)  # Avoid division by zero
        return {
            'data': torch.tensor(padded_data, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'rg': torch.tensor(rg, dtype=torch.float32),
            'code': code,
            'weight': torch.tensor(weight, dtype=torch.float32)
        }

# Load and preprocess data with per-sample normalization and modified augmentation
def load_data():
    xml_path = 'C:\\SASBDB\\sasbdb_data.xml'
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found at {xml_path}")
    tree = ET.parse(xml_path, parser=ET.XMLParser(encoding='utf-8'))
    root = tree.getroot()
    entries = root.findall('Entry')

    rg_dict = {}
    skipped_codes = []
    for entry in entries:
        code = entry.find('Code').text
        guinier_rg = entry.find('Guinier_Rg').text
        if guinier_rg is None or guinier_rg == 'None' or guinier_rg.strip() == '':
            skipped_codes.append((code, "Missing or invalid Rg"))
            logging.warning(f"Code {code}: Missing or invalid Rg")
            continue
        try:
            rg = float(guinier_rg)
            if rg <= 0 or rg > 100:  # Stricter Rg range
                skipped_codes.append((code, f"Invalid Rg range: {rg}"))
                logging.warning(f"Code {code}: has invalid Rg range ({rg}), skipped.")
                continue
            rg_dict[code] = rg
        except ValueError:
            skipped_codes.append((code, f"Invalid Rg value: {guinier_rg}"))
            logging.warning(f"Code {code}: Invalid Rg value ({guinier_rg})")
            continue

    logging.info(f"Loaded {len(rg_dict)} valid entries from XML")

    curve_dir = 'C:\\SASBDB\\curve'
    curve_data = {}
    aug_rg_dict = {}
    aug_curve_data = {}

    for code in list(rg_dict.keys()):
        file_path = os.path.join(curve_dir, f'{code}.txt')
        if not os.path.exists(file_path):
            skipped_codes.append((code, "Curve file missing"))
            logging.warning(f"Code {code}: Curve file missing")
            del rg_dict[code]
            continue
        try:
            data = np.loadtxt(file_path, usecols=(0, 1), dtype=np.float32)
     
            total_rows = len(data)
         
            valid_mask = ~np.any(np.isnan(data) | np.isinf(data) | (data < 0), axis=1)
            filtered_data = data[valid_mask]
            skipped_rows = total_rows - len(filtered_data)
            if skipped_rows > 0:
                logging.info(f"Code {code}: Skipped {skipped_rows} rows containing NaN, Inf, or negative I")

           
            if len(filtered_data) == 0:
                skipped_codes.append((code, "No valid data after filtering NaN, Inf, or negative I"))
                logging.info(f"Code {code}: Skipped due to no valid data after filtering NaN, Inf, or negative I")
                del rg_dict[code]
                continue

            rg = rg_dict[code]
            # Per-sample normalization for I only
            filtered_data[:, 1] = (filtered_data[:, 1] - filtered_data[:, 1].min()) / (filtered_data[:, 1].max() - filtered_data[:, 1].min() + 1e-10)
            curve_data[code] = filtered_data
            logging.info(f"Code {code}: Loaded {len(filtered_data)} valid rows")

            # Data augmentation for Rg > 0, only for I
            if rg > 0:
                for i in range(4):  # Generate 4 augmented samples
                    aug_code = f"{code}_aug_{i}"
                    aug_data = filtered_data.copy()
                    # Only add noise to I (column 1)
                    noise = np.random.normal(0, 0.003, aug_data[:, 1].shape).astype(np.float32)
                    aug_data[:, 1] = aug_data[:, 1] + noise
                    aug_data[:, 1] = np.clip(aug_data[:, 1], 0, None)
                    # Apply normalization to augmented I
                    aug_data[:, 1] = (aug_data[:, 1] - aug_data[:, 1].min()) / (aug_data[:, 1].max() - aug_data[:, 1].min() + 1e-10)
                    aug_curve_data[aug_code] = aug_data
                    aug_rg_dict[aug_code] = rg

        except Exception as e:
            skipped_codes.append((code, f"Error loading curve: {str(e)}"))
            logging.warning(f"Code {code}: Error loading curve ({str(e)})")
            del rg_dict[code]
            continue

    
    if not curve_data:
        logging.error(f"No valid data loaded. Total entries in XML: {len(entries)}, "
                      f"Valid Rg entries: {len(rg_dict)}, Skipped codes: {len(skipped_codes)}")
        for code, reason in skipped_codes:
            logging.error(f"Skipped {code}: {reason}")
        raise ValueError(
            "No valid data available after preprocessing. Check input files or filtering criteria in train.log.")

    curve_data.update(aug_curve_data)
    rg_dict.update(aug_rg_dict)

    lengths = [len(data) for data in curve_data.values()]
    max_length = int(np.percentile(lengths, 95))
    logging.info(f"Dataset loaded: {len(curve_data)} samples, max_length: {max_length}")
    logging.info(f"Skipped codes: {len(skipped_codes)}")
    for code, reason in skipped_codes:
        logging.info(f"Skipped {code}: {reason}")

    # Split into train and validation sets
    codes = list(curve_data.keys())
    np.random.shuffle(codes)
    train_size = int(0.8 * len(codes))
    train_codes = codes[:train_size]
    val_codes = codes[train_size:]

    train_curve_data = {code: curve_data[code] for code in train_codes}
    train_rg_dict = {code: rg_dict[code] for code in train_codes}
    val_curve_data = {code: curve_data[code] for code in val_codes}
    val_rg_dict = {code: rg_dict[code] for code in val_codes}

    train_dataset = SAXSDataset(train_curve_data, train_rg_dict, max_length)
    val_dataset = SAXSDataset(val_curve_data, val_rg_dict, max_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    return train_loader, val_loader, max_length

# Train the model with improved strategies
def train_model():
    train_loader, val_loader, max_length = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedSAXSModel(input_dim=2, d_model=512, nhead=16, num_layers=4, dim_feedforward=2048,
                              max_length=max_length, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.MSELoss(reduction='none')
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    num_epochs = 200

    output_dir = 'C:\\SASBDB\\model\\34'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_true_rgs = []
        train_pred_rgs = []
        for batch in train_loader:
            data = batch['data'].to(device)
            mask = batch['mask'].to(device)
            rg = batch['rg'].to(device)
            weights = batch['weight'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(data, mask)
                output = output.squeeze(-1)  # Fix shape issue
                loss = criterion(output, rg)
                weighted_loss = (loss * weights).mean()

            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += weighted_loss.item() * len(batch['rg'])

            # Collect predictions and true values
            pred_rg = output.detach().cpu().numpy()
            true_rg = rg.detach().cpu().numpy()
            train_pred_rgs.extend(pred_rg)
            train_true_rgs.extend(true_rg)

        train_loss /= len(train_loader.dataset)
        # Calculate training metrics
        train_true_rgs = np.array(train_true_rgs)
        train_pred_rgs = np.array(train_pred_rgs)
        train_rmse = np.sqrt(mean_squared_error(train_true_rgs, train_pred_rgs))
        train_mae = mean_absolute_error(train_true_rgs, train_pred_rgs)
        train_r2 = r2_score(train_true_rgs, train_pred_rgs)
        train_mre = np.mean(np.abs((train_true_rgs - train_pred_rgs) / train_true_rgs)) * 100

        # Log training metrics
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, "
                     f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, "
                     f"Train R²: {train_r2:.4f}, Train MRE: {train_mre:.2f}%")

        # Validation
        model.eval()
        val_loss = 0
        val_true_rgs = []
        val_pred_rgs = []
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(device)
                mask = batch['mask'].to(device)
                rg = batch['rg'].to(device)
                weights = batch['weight'].to(device)

                with torch.amp.autocast('cuda'):
                    output = model(data, mask)
                    output = output.squeeze(-1)  # Fix shape issue
                    loss = criterion(output, rg)
                    weighted_loss = (loss * weights).mean()

                val_loss += weighted_loss.item() * len(batch['rg'])

                # Collect predictions and true values
                pred_rg = output.detach().cpu().numpy()
                true_rg = rg.detach().cpu().numpy()
                val_pred_rgs.extend(pred_rg)
                val_true_rgs.extend(true_rg)

        val_loss /= len(val_loader.dataset)
        # Calculate validation metrics
        val_true_rgs = np.array(val_true_rgs)
        val_pred_rgs = np.array(val_pred_rgs)
        val_rmse = np.sqrt(mean_squared_error(val_true_rgs, val_pred_rgs))
        val_mae = mean_absolute_error(val_true_rgs, val_pred_rgs)
        val_r2 = r2_score(val_true_rgs, val_pred_rgs)
        val_mre = np.mean(np.abs((val_true_rgs - val_pred_rgs) / val_true_rgs)) * 100

        # Log validation metrics
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.6f}, "
                     f"Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, "
                     f"Val R²: {val_r2:.4f}, Val MRE: {val_mre:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)
        if scheduler.num_bad_epochs == 0 and scheduler.last_epoch > 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Learning rate reduced to {current_lr:.6f} at epoch {epoch + 1}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_transformer_model.pth'))
            logging.info("Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

if __name__ == '__main__':
    train_model()
