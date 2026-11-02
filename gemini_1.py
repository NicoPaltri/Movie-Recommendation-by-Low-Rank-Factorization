import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Caricamento Dati
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=columns)

# 2. Doppio Split (80% Train, 10% Validation, 10% Test)
# Primo split: separiamo il train (80%) dal resto (20%)
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)

# Secondo split: dividiamo il resto a metà (10% Val, 10% Test)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 3. Gestione Cold Start: Filtriamo sia Validation che Test
# Teniamo solo le righe dove utente E film esistono nel Training set
valid_users = train_data['user_id'].unique()
valid_items = train_data['item_id'].unique()

val_data = val_data[val_data['user_id'].isin(valid_users) & val_data['item_id'].isin(valid_items)]
test_data = test_data[test_data['user_id'].isin(valid_users) & test_data['item_id'].isin(valid_items)]

print(f"Dimensione Totale: {len(df)}")
print(f"Training Set: {len(train_data)} recensioni")
print(f"Validation Set: {len(val_data)} recensioni (filtrato)")
print(f"Test Set: {len(test_data)} recensioni (filtrato)")

from sklearn.metrics import mean_squared_error
# ==========================================
# STEP 2: CLASSE BASELINE (Milestone 2)
# ==========================================
class BaselineModel:
    def __init__(self):
        # Inizializziamo i parametri vuoti
        self.global_mu = 0.0
        self.b_i = {} # Dizionario per i bias dei film
        self.b_u = {} # Dizionario per i bias degli utenti

    def fit(self, train_df):
        """Addestra il modello calcolando le medie sul training set."""
        print("Calcolo delle Baseline in corso...")
        
        # 1. Calcolo Media Globale (mu)
        self.global_mu = train_df['rating'].mean()
        
        # 2. Calcolo Bias dei Film (b_i)
        # Differenza tra la media dei voti del singolo film e la media globale
        item_means = train_df.groupby('item_id')['rating'].mean()
        self.b_i = (item_means - self.global_mu).to_dict()
        
        # 3. Calcolo Bias degli Utenti (b_u)
        # Per calcolarlo correttamente, dobbiamo isolare l'effetto dell'utente
        # sottraendo dal suo voto sia la media globale che il bias del film
        train_df_copy = train_df.copy()
        
        # Mappiamo i bias dei film sul dataframe. Se un film non c'è, usiamo 0
        train_df_copy['b_i'] = train_df_copy['item_id'].map(self.b_i).fillna(0)
        
        # Calcoliamo il residuo: voto_reale - mu - b_i
        train_df_copy['rating_diff'] = train_df_copy['rating'] - self.global_mu - train_df_copy['b_i']
        
        # Il bias dell'utente è la media di questi residui
        user_means = train_df_copy.groupby('user_id')['rating_diff'].mean()
        self.b_u = user_means.to_dict()
        
        print("Calcolo completato.")

    def predict(self, u, i):
        """Stima il voto applicando la regola di FALLBACK per i dati mancanti."""
        # Si parte sempre dalla media globale
        pred = self.global_mu
        
        # Fallback Film: se il film è stato visto nel training, aggiungiamo il suo bias
        if i in self.b_i:
            pred += self.b_i[i]
            
        # Fallback Utente: se l'utente è presente nel training, aggiungiamo il suo bias
        if u in self.b_u:
            pred += self.b_u[u]
            
        # Clipping: ci assicuriamo che la predizione resti nella scala 1-5 stelle
        return max(1.0, min(5.0, pred))

    def evaluate(self, eval_df):
        """Calcola l'errore RMSE sul dataset fornito."""
        y_true = eval_df['rating'].values
        # Generiamo le predizioni per ogni riga usando la nostra funzione predict
        y_pred = [self.predict(row.user_id, row.item_id) for row in eval_df.itertuples(index=False)]
        
        # Root Mean Squared Error
        return np.sqrt(mean_squared_error(y_true, y_pred))

# ==========================================
# STEP 3: ESECUZIONE E VALUTAZIONE
# ==========================================
# Istanziamo e addestriamo il modello
baseline_model = BaselineModel()
baseline_model.fit(train_data)

# Calcoliamo l'RMSE sui vari set
train_rmse = baseline_model.evaluate(train_data)
val_rmse = baseline_model.evaluate(val_data)
test_rmse = baseline_model.evaluate(test_data)

print("\n--- RISULTATI BASELINE (Benchmark) ---")
print(f"RMSE Training Set:   {train_rmse:.4f}")
print(f"RMSE Validation Set: {val_rmse:.4f}")
print(f"RMSE Test Set:       {test_rmse:.4f}")



