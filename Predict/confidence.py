import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

# Define the model class (same as provided)
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

# Define dataset class for prediction
class SAXSPredictDataset(Dataset):
    def __init__(self, curve_data, max_length):
        self.curve_data = curve_data
        self.max_length = max_length
        self.filenames = list(curve_data.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = self.curve_data[filename]
        padded_data = np.zeros((self.max_length, 2), dtype=np.float32)
        seq_len = min(len(data), self.max_length)
        padded_data[:seq_len, :] = data[:seq_len, :]
        mask = np.zeros(self.max_length, dtype=np.float32)
        mask[:seq_len] = 1
        return {
            'data': torch.tensor(padded_data, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'filename': filename
        }

# Load and preprocess new data
def load_new_data(data_dir, max_length):
    curve_data = {}
    for i in range(23):
        filename = f'S_PG-L-T{i}.dat'
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found, skipping.")
            continue
        try:
            # Load data, skipping header lines starting with '#'
            data = np.loadtxt(file_path, usecols=(0, 1), dtype=np.float32, comments='#')
            
            data[:, 0] = data[:, 0]
            # Normalize intensity (I) to [0, 1]
            data[:, 1] = (data[:, 1] ) / (data[:, 1].max() + 1e-10)
            # Filter out invalid data
            valid_mask = ~np.any(np.isnan(data) | np.isinf(data) | (data < 0), axis=1)
            filtered_data = data[valid_mask]
            if len(filtered_data) == 0:
                print(f"File {filename}: No valid data after filtering.")
                continue
            curve_data[filename] = filtered_data
            print(f"File {filename}: Loaded {len(filtered_data)} valid rows.")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue

    dataset = SAXSPredictDataset(curve_data, max_length)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return data_loader

def predict_with_uncertainty(model, data, mask, n_iter=100):
    """
    Monte Carlo Dropout uncertainty estimation

    Returns:
        mean_pred : mean prediction
        std_pred  : prediction uncertainty
    """

    preds = []

    model.train()  

    with torch.no_grad():
        for _ in range(n_iter):
            output = model(data, mask)
            preds.append(
                output.squeeze(-1).cpu().numpy()
            )

    preds = np.stack(preds, axis=0)

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    return mean_pred, std_pred

# Predict Rg values
def predict():
    data_dir = r'C:\Users\zhang\Desktop\scattering curve\PG'
    max_length = 2544  # As per original code
    data_loader = load_new_data(data_dir, max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedSAXSModel(input_dim=2, d_model=512, nhead=16, num_layers=4, dim_feedforward=2048,
                              max_length=max_length, dropout=0.2).to(device)
    model.load_state_dict(torch.load('D:/SASBDB/model/best_transformer_model.pth'))
    model.eval()

    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Rg.txt')
    predictions = []

    for batch in data_loader:

        data = batch['data'].to(device)
        mask = batch['mask'].to(device)
        filenames = batch['filename']

        with torch.amp.autocast('cuda'):

            pred_rgs, pred_stds = predict_with_uncertainty(
                model,
                data,
                mask,
                n_iter=100
            )

        for filename, pred_rg, pred_std in zip(
                filenames,
                pred_rgs,
                pred_stds):
            ci_lower = pred_rg - 1.96 * pred_std
            ci_upper = pred_rg + 1.96 * pred_std

            relative_std = pred_std / (abs(pred_rg) + 1e-8)

            confidence = np.exp(
                -5 * relative_std
            ) * 100

            predictions.append(
                (
                    filename,
                    pred_rg,
                    pred_std,
                    confidence,
                    ci_lower,
                    ci_upper
                )
            )

    with open(output_file, 'w') as f:

        f.write(
            "Filename\t"
            "Predicted_Rg\t"
            "Std\t"
            "Confidence(%)\t"
            "CI_lower\t"
            "CI_upper\n"
        )

        for filename, rg, std, conf, low, high in predictions:
            f.write(
                f"{filename}\t"
                f"{rg:.4f}\t"
                f"{std:.4f}\t"
                f"{conf:.2f}\t"
                f"{low:.4f}\t"
                f"{high:.4f}\n"
            )

if __name__ == '__main__':
    predict()