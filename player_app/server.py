from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_def import PlayerRatingLSTM
import os

app = Flask(__name__)

df = pd.read_csv('player_attributes_cleaned.csv')
feature_cols = [col for col in df.columns if col not in ['overall_rating', 'player_api_id', 'date', 'player_name']]
sequence_length = 3

model = PlayerRatingLSTM(40)
model.load_state_dict(torch.load("player_nn_model_weights.pth"))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    player = request.form['player']
    player_df = df[df['player_name'] == player].reset_index(drop=True)

    if player_df.empty or len(player_df) < sequence_length:
        return render_template('result.html', player=player, image_path=None,
                               error="Player not found or not enough data.")

    try:
        X_player = []
        for i in range(len(player_df) - sequence_length):
            X_seq = player_df.loc[i:i+sequence_length-1, feature_cols].to_numpy(dtype=np.float32)
            X_player.append(X_seq)

        X_tensor = torch.tensor(np.stack(X_player))
        with torch.no_grad():
            preds = model(X_tensor).numpy().flatten()

        actual = player_df['overall_rating'].iloc[sequence_length:].to_numpy()
        ages = player_df['age'].iloc[sequence_length:].to_numpy()
        mse = np.mean((actual - preds) ** 2)

        # 🔥 Plot and save to static/plot.png
        plt.figure(figsize=(10, 5))
        plt.plot(ages, actual, label='Actual', marker='o')
        plt.plot(ages, preds, label='Predicted', marker='x')
        for age, a, p in zip(ages, actual, preds):
            plt.plot([age, age], [a, p], color='gray', linestyle='--', linewidth=0.7)
        plt.xlabel('Age')
        plt.ylabel('Overall Rating')
        plt.ylim([30, 100])
        plt.title(f'{player} - MSE: {mse:.2f}')
        plt.legend()

        plot_path = os.path.join('static', 'plot.svg')
        plt.savefig(plot_path)
        plt.close()

        return render_template('result.html', player=player, image_path=plot_path, error=None)

    except Exception as e:
        return render_template('result.html', player=player, image_path=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)