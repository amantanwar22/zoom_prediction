### Pipeline Overview:

## 1 In line 5 of main.py we have function load_and_normalize()
Reads both logs, merges on timestamp, normalizes latency_ms + packet_loss.

## 2 In line 40 we have function train_autoencoder()
Autoencoder learns baseline pattern using reconstruction loss.

## 3 In line 57 we have function detect_anomalies()
High reconstruction error > quantile threshold ⇒ anomaly flagged.

## 4 In line 87 we have function train_zoom_model()
NN predicts zoom_freeze_count and call_rating.

## 5 In line 102 we have function predict_zoom_quality()
Compares true vs predicted metrics for all timestamps.

### WorkFLow:

## 1 
So first made an unsupervised anomaly detection model using an Autoencoder(line 22) which learns what normal network is and if there is high reconstruction error it flags it as an anomaly.

## 2 
Then created supervised model (line 73) which which predicts zoom freeze_count and call_rating based on network's latency and packet_loss.

## 3 
Now we can detect both network abnormal behaviour and how it impacts zoom quality.
