import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from torch.amp import autocast
import time


logging.basicConfig(filename='C:/SASBDB/model/33/predict.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
        self.residual_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
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
        residual = x
        x = self.residual_block(x) + residual
        x = self.mlp(x)
        return x


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


def load_data():
    xml_path = 'C:/SASBDB/sasbdb_data.xml'
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
            if rg <= 0 or rg > 100:
                skipped_codes.append((code, f"Invalid Rg range: {rg}"))
                logging.warning(f"Code {code}: has invalid Rg range ({rg}), skipped.")
                continue
            rg_dict[code] = rg
        except ValueError:
            skipped_codes.append((code, f"Invalid Rg value: {guinier_rg}"))
            logging.warning(f"Code {code}: Invalid Rg value ({guinier_rg})")
            continue

    logging.info(f"Loaded {len(rg_dict)} valid entries from XML")

    curve_dir = 'C:/SASBDB/curve'
    curve_data = {}
    load_times = []
    load_codes = []
    for code in list(rg_dict.keys()):
        file_path = os.path.join(curve_dir, f'{code}.txt')
        if not os.path.exists(file_path):
            skipped_codes.append((code, "Curve file missing"))
            logging.warning(f"Code {code}: Curve file missing")
            continue
        try:
            start_time = time.perf_counter()  
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
                continue

            filtered_data[:, 1] = (filtered_data[:, 1] - filtered_data[:, 1].min()) / (filtered_data[:, 1].max() - filtered_data[:, 1].min() + 1e-10)
            curve_data[code] = filtered_data
            end_time = time.perf_counter()
            load_time = end_time - start_time
            load_times.append(load_time)
            load_codes.append(code)
            logging.info(f"Code {code}: Loaded {len(filtered_data)} valid rows in {load_time:.9f} seconds")
        except Exception as e:
            skipped_codes.append((code, f"Error loading curve: {str(e)}"))
            logging.warning(f"Code {code}: Error loading curve: {str(e)}")
            continue

    lengths = [len(data) for data in curve_data.values()]
    max_length = 2544
    logging.info(f"Dataset loaded: {len(curve_data)} samples, max_length: {max_length}")
    logging.info(f"Skipped codes: {len(skipped_codes)}")
    for code, reason in skipped_codes:
        logging.info(f"Skipped {code}: {reason}")

    total_load_time = sum(load_times)
    avg_load_time = total_load_time / len(load_times) if load_times else 0
    with open(os.path.join('C:/SASBDB/model/33', 'load_times.txt'), 'w') as f:
        f.write("Code\tLoad_Time_s\n")
        for code, load_time in zip(load_codes, load_times):
            f.write(f"{code}\t{load_time:.9f}\n")
        f.write(f"\nTotal Load Time: {total_load_time:.9f} seconds\n")
        f.write(f"Average Load Time: {avg_load_time:.9f} seconds\n")
    logging.info(f"Load times saved to {os.path.join('C:/SASBDB/model/33', 'load_times.txt')}")

    dataset = SAXSDataset(curve_data, rg_dict, max_length)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return data_loader, max_length


def predict_and_draw():
    data_loader, max_length = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedSAXSModel(input_dim=2, d_model=512, nhead=16, num_layers=4, dim_feedforward=2048,
                              max_length=max_length, dropout=0.2).to(device)
    model.load_state_dict(torch.load('C:/SASBDB/model/33/best_transformer_model.pth'))
    model.eval()

    output_dir = 'C:/SASBDB/model/33'
    os.makedirs(output_dir, exist_ok=True)

    true_rgs = []
    pred_rgs = []
    codes = []
    prediction_times = []
    transfer_times = []

    with torch.no_grad():
        for batch in data_loader:
            data = batch['data']
            mask = batch['mask']
            rg = batch['rg'].to(device)
            code = batch['code']


            start_transfer = time.perf_counter()
            data = data.to(device)
            mask = mask.to(device)
            transfer_time = time.perf_counter() - start_transfer
            batch_size = len(code)
            per_sample_transfer_time = transfer_time / batch_size
            transfer_times.extend([per_sample_transfer_time] * batch_size)


            start_time = time.perf_counter()
            with torch.amp.autocast('cuda'):
                output = model(data, mask)
                output = output.squeeze(-1)
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            per_sample_time = batch_time / batch_size
            prediction_times.extend([per_sample_time] * batch_size)

            pred_rg = output.cpu().numpy()
            true_rg = rg.cpu().numpy()
            pred_rgs.extend(pred_rg)
            true_rgs.extend(true_rg)
            codes.extend(code)

    true_rgs = np.array(true_rgs)
    pred_rgs = np.array(pred_rgs)


    rmse = np.sqrt(mean_squared_error(true_rgs, pred_rgs))
    mae = mean_absolute_error(true_rgs, pred_rgs)
    r2 = r2_score(true_rgs, pred_rgs)
    mre = np.mean(np.abs((true_rgs - pred_rgs) / true_rgs)) * 100


    logging.info(f"Final Metrics - RMSE: {rmse:.4f} nm, MAE: {mae:.4f} nm, R²: {r2:.4f}, Mean Relative Error: {mre:.2f}%")


    results_df = pd.DataFrame({
        'Code': codes,
        'True_Rg': true_rgs,
        'Predicted_Rg': pred_rgs,
        'Prediction_Time_s': prediction_times
    })
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    logging.info(f"Predictions saved to {os.path.join(output_dir, 'predictions.csv')}")


    total_time = sum(prediction_times)
    avg_time = total_time / len(prediction_times) if prediction_times else 0
    with open(os.path.join(output_dir, 'prediction_times.txt'), 'w') as f:
        f.write("Code\tPrediction_Time_s\n")
        for code, pred_time in zip(codes, prediction_times):
            f.write(f"{code}\t{pred_time:.9f}\n")
        f.write(f"\nTotal Prediction Time: {total_time:.9f} seconds\n")
        f.write(f"Average Prediction Time: {avg_time:.9f} seconds\n")
    logging.info(f"Timing information saved to {os.path.join(output_dir, 'prediction_times.txt')}")


    total_transfer_time = sum(transfer_times)
    avg_transfer_time = total_transfer_time / len(transfer_times) if transfer_times else 0
    with open(os.path.join(output_dir, 'transfer_times.txt'), 'w') as f:
        f.write("Code\tTransfer_Time_s\n")
        for code, transfer_time in zip(codes, transfer_times):
            f.write(f"{code}\t{transfer_time:.9f}\n")
        f.write(f"\nTotal Transfer Time: {total_transfer_time:.9f} seconds\n")
        f.write(f"Average Transfer Time: {avg_transfer_time:.9f} seconds\n")
    logging.info(f"Transfer timing information saved to {os.path.join(output_dir, 'transfer_times.txt')}")

if __name__ == '__main__':
    predict_and_draw()
