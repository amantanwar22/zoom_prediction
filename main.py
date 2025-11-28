import pandas as pd
import torch
import torch.nn as nn

def load_and_normalize():
    network = pd.read_csv("network_logs.csv")
    zoom = pd.read_csv("zoom_logs.csv")
    merged = pd.merge(network,zoom, on ="timestamp", how="inner")

    network_features = merged[["latency_ms","packet_loss"]].values.astype("float32")
    data = torch.tensor(network_features)

    mins = data.min(dim=0).values
    maxs = data.max(dim=0).values
    range_ = maxs - mins
    range_[range_==0]=1.0

    data_normalised = (data - mins) / range_
    return merged,data_normalised


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(data_normalised,epochs=2000):
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    normal_data = data_normalised[:3]

    for _ in range(epochs):
        output = model(normal_data)
        loss = loss_fn(output, normal_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def detect_anomalies(model,data_normalised):
    with torch.no_grad():
        reconstructed = model(data_normalised)
        errors = torch.mean((reconstructed - data_normalised)**2,dim=1)


    threshold = errors.quantile(0.75)
    is_anomaly = errors > threshold
    return errors,threshold,is_anomaly


# already done -> Detect anomalies in network
#Now predicting zoom quality given the network
# zoom_targets = merged[["zoom_freeze_count","call_rating"]].values.astype("float32")
# y = torch.tensor(zoom_targets)

class ZoomPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self,x):
        return self.net(x)

def train_zoom_model(x,y,epochs=2000):
    zoom_model = ZoomPredictor()
    zoom_optimizer = torch.optim.Adam(zoom_model.parameters(), lr=0.01)
    zoom_loss_fn = nn.MSELoss()

    for _ in range(epochs):
        pred = zoom_model(x)
        loss_zoom = zoom_loss_fn(pred,y)

        zoom_optimizer.zero_grad()
        loss_zoom.backward()
        zoom_optimizer.step()

    return zoom_model

def predict_zoom_quality(zoom_model,merged,x):
    with torch.no_grad():
        zoom_pred = zoom_model(x)

    print("ZOOM QUALITY PREDICTIONS")
    for i in range(len(merged)):
        ts = merged.iloc[i]["timestamp"]
        true_freeze = merged.iloc[i]["zoom_freeze_count"]
        true_rating = merged.iloc[i]["call_rating"]

        pred_freeze = zoom_pred[i][0].item()
        pred_rating = zoom_pred[i][1].item()
        print(
            f"{ts} | true_freezes={true_freeze}  pred_freezes={pred_freeze:.2f}  "
            f"| true_rating={true_rating}  pred_rating={pred_rating:.2f}"
        )

if __name__ == "__main__":

    merged, data_normalised = load_and_normalize()

    auto_model = train_autoencoder(data_normalised)

    errors, threshold, anomalies = detect_anomalies(auto_model, data_normalised)
    print("ANOMALY REPORT")
    for i in range(len(errors)):
        row = merged.iloc[i]
        status = "ANOMALY" if anomalies[i] else "NORMAL"
        print(f"{row['timestamp']} | error={errors[i].item():.4f} --> {status}")

    zoom_targets = merged[["zoom_freeze_count","call_rating"]].values.astype("float32")
    y = torch.tensor(zoom_targets)
    zoom_model = train_zoom_model(data_normalised, y)
    predict_zoom_quality(zoom_model, merged, data_normalised)


