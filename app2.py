import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
import os

# Set page config
st.set_page_config(page_title="Chess Opening Recommendation System", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Chess_pieces_close_up.jpg/1200px-Chess_pieces_close_up.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: scroll;
        text-color: white;
        font-weight: 600;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
# Title and description
st.title("Chess Opening Recommendation System")
st.write("Get personalized chess opening recommendations based on your preferences")

# Function to load the dataset
@st.cache_data
def load_data():
    try:
        chess = pd.read_csv("games.csv")
        # Remove duplicates based on id
        chess = chess.drop_duplicates(subset=['id'])
        
        # Extract opening archetype
        chess = chess.assign(
            opening_archetype=chess.opening_name.map(
                lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()
            ),
            opening_moves=chess.apply(lambda srs: ' '.join(srs['moves'].split(" ")[:srs['opening_ply']]),
                                  axis=1)
        )
        return chess
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to load models
@st.cache_resource
def load_models():
    try:
        # Load content-based model
        with open(r'models/content_based_model.pkl', 'rb') as f:
            content_based_model = pickle.load(f)
        
        # Load collaborative data
        with open(r'models/collaborative_data.pkl', 'rb') as f:
            collaborative_data = pickle.load(f)
        
        # Load collaborative model
        collaborative_model = {
            'model': tf.keras.models.load_model(r'models/collaborative_model.keras')
        }
        
        # Load additional collaborative model data
        with open(r'models/collaborative_model_data.pkl', 'rb') as f:
            collab_data = pickle.load(f)
            collaborative_model.update(collab_data)
        
        # Load hybrid model
        with open(r'models/hybrid_model.pkl', 'rb') as f:
            hybrid_model = pickle.load(f)
        
        return content_based_model, collaborative_data, collaborative_model, hybrid_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training notebook first to generate the models.")
        return None, None, None, None

# Function to get content-based recommendations
def get_content_based_recommendations(favorite_openings, content_based_model, top_n=5):
    """
    Mendapatkan rekomendasi berdasarkan opening favorit dengan model content-based
    """
    similarity_df = content_based_model['similarity_matrix']
    opening_names = content_based_model['opening_names']
    
    # Filter rekomendasi hanya untuk opening yang ada dalam dataset
    valid_favorites = [opening for opening in favorite_openings if opening in opening_names]
    
    if not valid_favorites:
        return pd.DataFrame(columns=['opening_name', 'similarity_score'])
        
    # Hitung rata-rata similarity scores untuk semua opening favorit
    similarity_scores = similarity_df[valid_favorites].mean(axis=1)
    
    # Buat DataFrame rekomendasi
    recommendations = pd.DataFrame({
        'opening_name': similarity_scores.index,
        'similarity_score': similarity_scores.values
    })
    
    def calibrate_score(score, target_min=0.5, target_max=0.99):
        """
        Mengkalibrasi skor agar memiliki rentang yang diinginkan.
        """
        # Normalisasi skor ke rentang 0 hingga 1
        score_norm = (score - np.min(score)) / (np.max(score) - np.min(score))
        
        # Skalakan skor ke rentang yang diinginkan
        calibrated_score = score_norm * (target_max - target_min) + target_min
        
        return calibrated_score

    # Apply softmax to the similarity scores
    recommendations['similarity_score'] = calibrate_score(recommendations['similarity_score'])
    
    # Filter out favorite openings
    recommendations = recommendations[~recommendations.opening_name.isin(favorite_openings)]
    
    # Sort berdasarkan similarity score
    recommendations = recommendations.sort_values('similarity_score', ascending=False)
    
    return recommendations.head(top_n)
        
# Function to get collaborative filtering recommendations
def get_collaborative_recommendations(user_rating, collaborative_data, collaborative_model, top_n=5, debug=True):
    """
    Mendapatkan rekomendasi opening berdasarkan rating pemain dengan pendekatan yang lebih sensitif terhadap rating.

    Parameters:
    -----------
    user_rating : int
        Rating pemain yang meminta rekomendasi
    collaborative_data : dict
        Dictionary berisi data untuk collaborative filtering
    collaborative_model : dict
        Dictionary berisi model collaborative filtering
    top_n : int
        Jumlah rekomendasi yang ingin ditampilkan
    debug : bool
        Aktifkan mode debug untuk melihat proses detail

    Returns:
    --------
    DataFrame
        Dataframe berisi rekomendasi opening dengan skor
    """

    def softmax(x):
        """
        Fungsi softmax yang membuat distribusi probabilitas dari array nilai.
        Lebih baik untuk mempertahankan perbedaan relatif antara nilai prediksi.
        """
        # Kurangi max(x) untuk stabilitas numerik
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


    def estimate_opening_complexity(collaborative_data):
        """
        Estimasi kompleksitas pembukaan berdasarkan rating rata-rata pemain yang menggunakannya.
        Pembukaan dengan rating rata-rata lebih tinggi dianggap lebih kompleks.

        Returns:
        --------
        Series
            Series berisi estimasi kompleksitas untuk setiap pembukaan
        """
        # Ambil matriks pemain-pembukaan
        player_opening = collaborative_data['player_opening_matrix'].copy()

        # Gabungkan dengan data pemain untuk mendapatkan rating
        player_data = collaborative_data['player_data']
        player_opening = player_opening.merge(player_data[['player_id', 'rating']], on='player_id', how='left')

        # Hitung rating rata-rata per pembukaan
        opening_avg_rating = player_opening.groupby('opening_name')['rating'].mean()

        # Normalisasi ke [0, 1]
        if opening_avg_rating.max() - opening_avg_rating.min() > 0:
            normalized_complexity = (opening_avg_rating - opening_avg_rating.min()) / (opening_avg_rating.max() - opening_avg_rating.min())
        else:
            normalized_complexity = pd.Series(0.5, index=opening_avg_rating.index)

        return normalized_complexity


    def adjust_by_rating(predictions, complexity_scores, user_rating, rating_max=3000, influence=0.3):
        """
        Menyesuaikan prediksi berdasarkan rating pengguna dan kompleksitas pembukaan.

        Parameters:
        -----------
        predictions : Series
            Skor prediksi untuk setiap pembukaan
        complexity_scores : Series
            Skor kompleksitas untuk setiap pembukaan
        user_rating : int
            Rating pengguna
        rating_max : int
            Rating maksimum yang diperkirakan (untuk normalisasi)
        influence : float
            Seberapa besar pengaruh rating (0-1)

        Returns:
        --------
        Series
            Prediksi yang disesuaikan
        """
        # Normalisasi rating pengguna
        normalized_rating = user_rating / rating_max

        # Hitung faktor penyesuaian
        # Pemain dengan rating rendah mendapat boost untuk pembukaan sederhana
        # Pemain dengan rating tinggi mendapat boost untuk pembukaan kompleks
        common_openings = set(predictions.index) & set(complexity_scores.index)

        adjusted_predictions = predictions.copy()

        for opening in common_openings:
            complexity = complexity_scores.get(opening, 0.5)

            # Rating tinggi → lebih suka pembukaan kompleks
            # Rating rendah → lebih suka pembukaan sederhana
            rating_factor = normalized_rating - 0.5  # -0.5 to 0.5

            # Kompleksitas tinggi → kompleks, Kompleksitas rendah → sederhana
            complexity_factor = complexity - 0.5  # -0.5 to 0.5

            # Jika keduanya cocok (rating tinggi & kompleks ATAU rating rendah & sederhana),
            # maka berikan boost
            adjustment = influence * rating_factor * complexity_factor * 2  # * 2 untuk memperkuat efek

            adjusted_predictions[opening] = adjusted_predictions[opening] * (1 + adjustment)

        # Renormalisasi hasil
        if adjusted_predictions.max() > 0:
            adjusted_predictions = adjusted_predictions / adjusted_predictions.max()

        return adjusted_predictions

    if debug:
        print(f"Input user rating: {user_rating}")

    # Ambil data yang diperlukan
    player_data = collaborative_data['player_data']
    player_encoder = collaborative_data['player_encoder']
    opening_encoder = collaborative_data['opening_encoder']
    model = collaborative_model['model']

    # === IMPROVED: Cari pemain dengan rating yang mirip dengan user_rating dengan pembobotan ===
    rating_diff = abs(player_data['rating'] - user_rating)

    # Gunakan rating_range dinamis untuk memastikan kita mendapatkan pemain yang cukup
    min_similar_players = 10  # Meningkatkan jumlah minimum pemain serupa
    rating_ranges = [50, 100, 200, 300, 400, 500, 750, 1000]  # Range yang lebih halus

    similar_players_idx = None
    used_rating_range = None

    for rating_range in rating_ranges:
        similar_players_idx = rating_diff[rating_diff <= rating_range].index
        if len(similar_players_idx) >= min_similar_players:
            used_rating_range = rating_range
            break

    # Jika masih kurang, ambil pemain dengan rating terdekat
    if len(similar_players_idx) < min_similar_players:
        similar_players_idx = rating_diff.nsmallest(min_similar_players).index
        used_rating_range = "adaptive"

    # === IMPROVED: Terapkan pembobotan berdasarkan kedekatan rating ===
    # Pemain dengan rating yang lebih dekat akan memiliki pengaruh lebih besar
    rating_weights = 1 / (rating_diff.loc[similar_players_idx] + 10)  # +10 untuk menghindari pembagian dengan nol atau bobot terlalu besar

    # Normalisasi bobot agar jumlahnya 1
    rating_weights = rating_weights / rating_weights.sum()

    similar_players = player_data.loc[similar_players_idx, 'player_id'].unique()

    if debug:
        print(f"Found {len(similar_players)} similar players with rating around {user_rating} (range: {used_rating_range})")
        player_ratings = player_data.loc[similar_players_idx, 'rating'].values
        print(f"Similar players ratings: min={player_ratings.min()}, max={player_ratings.max()}, mean={player_ratings.mean():.1f}")
        print(f"Rating weights: min={rating_weights.min():.4f}, max={rating_weights.max():.4f}")

    # Filter similar players yang ada dalam encoder
    valid_similar_players = [p for p in similar_players if p in player_encoder.classes_]
    valid_player_indices = [similar_players_idx[i] for i, p in enumerate(similar_players) if p in player_encoder.classes_]

    # Sesuaikan bobot untuk pemain yang valid
    if valid_player_indices:
        valid_weights = rating_weights.loc[valid_player_indices]
        valid_weights = valid_weights / valid_weights.sum()  # Renormalisasi bobot
    else:
        valid_weights = None

    if debug:
        print(f"Found {len(valid_similar_players)} valid similar players in encoder")

    # Jika tidak ada pemain yang valid, gunakan pendekatan popularitas
    if len(valid_similar_players) == 0:
        if debug:
            print("Warning: No valid similar players in encoder, using popularity-based approach")

        opening_counts = collaborative_data['player_opening_matrix'].groupby('opening_name').size()

        # === IMPROVED: Terapkan faktor rating ke pendekatan popularitas ===
        # Ambil faktor kompleksitas pembukaan (kita akan estimasi berdasarkan distribusi rating)
        opening_complexity = estimate_opening_complexity(collaborative_data)

        # Sesuaikan popularitas berdasarkan rating
        adjusted_counts = adjust_by_rating(opening_counts, opening_complexity, user_rating)

        # Normalisasi
        normalized_counts = adjusted_counts / adjusted_counts.max() if adjusted_counts.max() > 0 else adjusted_counts

        recommendations = pd.DataFrame({
            'opening_name': adjusted_counts.index,
            'score': normalized_counts.values
        }).sort_values('score', ascending=False)

    else:  # Jika ada pemain yang valid
        # ========== IMPLEMENTASI LOGIKA REKOMENDASI YANG DITINGKATKAN ==========
        # 1. Encode players dan openings
        encoded_players = player_encoder.transform(valid_similar_players)
        encoded_openings = np.arange(len(opening_encoder.classes_))

        # 2. Buat input batch untuk prediksi
        players_batch = np.repeat(encoded_players, len(encoded_openings))
        openings_batch = np.tile(encoded_openings, len(encoded_players))

        # 3. Prediksi rating untuk semua kombinasi
        predictions = model.predict(
            [players_batch, openings_batch],
            batch_size=1024,
            verbose=0
        ).flatten()

        # 4. Reshape ke bentuk (num_players, num_openings)
        predictions_matrix = predictions.reshape(len(encoded_players), -1)

        # === IMPROVED: Gunakan pembobotan untuk rata-rata prediksi ===
        if valid_weights is not None:
            # Terapkan bobot ke prediksi
            weighted_predictions = np.zeros(predictions_matrix.shape[1])

            for i in range(len(valid_similar_players)):
                weighted_predictions += predictions_matrix[i] * valid_weights.iloc[i]

            avg_predictions = weighted_predictions
        else:
            # Fallback ke rata-rata biasa jika tidak ada bobot
            avg_predictions = np.mean(predictions_matrix, axis=0)

        if debug:
            print(f"Raw predictions (sample): {avg_predictions[:5]}")
            print(f"Min/Max raw prediction: {np.min(avg_predictions):.4f}/{np.max(avg_predictions):.4f}")

        # === IMPROVED: Gunakan softmax untuk normalisasi yang lebih sensitif ===
        # Softmax menjaga perbedaan relatif lebih baik daripada normalisasi min-max
        temperature = 2.0  # Parameter untuk mengontrol "kecuraman" distribusi softmax
        softmax_predictions = softmax(avg_predictions / temperature)

        # === IMPROVED: Sesuaikan prediksi berdasarkan kompleksitas pembukaan ===
        opening_complexity = estimate_opening_complexity(collaborative_data)

        # Sesuaikan prediksi berdasarkan rating user dan kompleksitas pembukaan
        final_predictions = adjust_by_rating(
            pd.Series(softmax_predictions, index=opening_encoder.classes_),
            opening_complexity,
            user_rating
        )

        if debug:
            print(f"After softmax normalization and rating adjustment (sample): {final_predictions[:5]}")
            print(f"Min/Max final prediction: {final_predictions.min():.4f}/{final_predictions.max():.4f}")

        # 7. Buat DataFrame rekomendasi
        recommendations = pd.DataFrame({
            'opening_name': final_predictions.index,
            'score': final_predictions.values
        }).sort_values('score', ascending=False)

        if debug:
            print("Collaborative filtering recommendations generated successfully")
            print(f"Top 5 recommended openings with scores:")
            for idx, row in recommendations.head().iterrows():
                print(f"  {row['opening_name']}: {row['score']:.4f}")

    return recommendations.head(top_n)

# Function to get hybrid recommendations
def get_hybrid_recommendations(favorite_openings, user_rating, 
                              content_based_model, collaborative_data, collaborative_model,
                              alpha=0.75, top_n=5):
    """
    Mendapatkan rekomendasi hybrid filtering dengan menggabungkan content-based dan collaborative filtering
    dengan normalisasi skor yang tepat dan debugging
    """
    # Get content-based recommendations with more results to ensure good coverage
    content_recs = get_content_based_recommendations(favorite_openings, content_based_model, top_n=50)
    
    # Get collaborative filtering recommendations with more results
    collab_recs = get_collaborative_recommendations(user_rating, collaborative_data, collaborative_model, top_n=50)
    
    # Debug output
    print(f"Content-based recommendations count: {len(content_recs)}")
    if len(content_recs) > 0:
        print(f"Content-based score range: {content_recs['similarity_score'].min()} to {content_recs['similarity_score'].max()}")
    
    print(f"Collaborative recommendations count: {len(collab_recs)}")
    if len(collab_recs) > 0:
        print(f"Collaborative score range: {collab_recs['score'].min()} to {collab_recs['score'].max()}")
    
    # If either recommendation is empty, prepare a properly formatted DataFrame before returning
    if len(content_recs) == 0:
        # Add the required columns to collaborative recommendations
        result_df = collab_recs.head(top_n).rename(columns={'score': 'cf_score'})
        result_df['cb_score'] = 0.0
        result_df['hybrid_score'] = (1 - alpha) * result_df['cf_score']  # Weight by (1-alpha) since it's only CF
        return result_df[['opening_name', 'hybrid_score', 'cb_score', 'cf_score']]
    elif len(collab_recs) == 0:
        # Add the required columns to content-based recommendations
        result_df = content_recs.head(top_n).rename(columns={'similarity_score': 'cb_score'})
        result_df['cf_score'] = 0.0
        result_df['hybrid_score'] = alpha * result_df['cb_score']  # Weight by alpha since it's only CB
        return result_df[['opening_name', 'hybrid_score', 'cb_score', 'cf_score']]
    
    # Copy DataFrames to avoid modifying originals
    content_recs = content_recs.copy()
    collab_recs = collab_recs.copy()
    
    # Rename scores for clarity and consistency
    content_recs['cb_score'] = content_recs['similarity_score']
    collab_recs['cf_score'] = collab_recs['score']
    
    # Find common openings between the two recommendation methods
    common_openings = set(content_recs['opening_name']) & set(collab_recs['opening_name'])
    print(f"Number of common openings: {len(common_openings)}")
    
    # Get only the openings and scores we need
    content_recs_subset = content_recs[['opening_name', 'cb_score']]
    collab_recs_subset = collab_recs[['opening_name', 'cf_score']]
    
    # Merge the recommendations
    hybrid_recs = pd.merge(content_recs_subset, collab_recs_subset, on='opening_name', how='outer')
    
    # Fill missing scores with minimum values instead of 0
    if len(content_recs) > 0:
        min_cb_score = content_recs['cb_score'].min()
        # Use a small fraction of the minimum as the fill value
        cb_fill_value = min_cb_score * 0.5 if min_cb_score > 0 else 0
    else:
        cb_fill_value = 0
        
    if len(collab_recs) > 0:
        min_cf_score = collab_recs['cf_score'].min()
        # Use a small fraction of the minimum as the fill value
        cf_fill_value = min_cf_score * 0.5 if min_cf_score > 0 else 0
    else:
        cf_fill_value = 0
    
    # Fill missing values with the calculated fill values
    hybrid_recs['cb_score'] = hybrid_recs['cb_score'].fillna(cb_fill_value)
    hybrid_recs['cf_score'] = hybrid_recs['cf_score'].fillna(cf_fill_value)
    
    # Normalize scores within their own ranges
    # Min-max normalization for content-based scores
    cb_min = hybrid_recs['cb_score'].min()
    cb_max = hybrid_recs['cb_score'].max()
    if cb_max > cb_min:
        hybrid_recs['cb_score_norm'] = (hybrid_recs['cb_score'] - cb_min) / (cb_max - cb_min)
    else:
        hybrid_recs['cb_score_norm'] = hybrid_recs['cb_score'] / cb_max if cb_max > 0 else 0

    # Min-max normalization for collaborative scores
    cf_min = hybrid_recs['cf_score'].min()
    cf_max = hybrid_recs['cf_score'].max()
    if cf_max > cf_min:
        hybrid_recs['cf_score_norm'] = (hybrid_recs['cf_score'] - cf_min) / (cf_max - cf_min)
    else:
        hybrid_recs['cf_score_norm'] = hybrid_recs['cf_score'] / cf_max if cf_max > 0 else 0
    
    # Print normalized score ranges for debugging
    print(f"Normalized CB score range: {hybrid_recs['cb_score_norm'].min()} to {hybrid_recs['cb_score_norm'].max()}")
    print(f"Normalized CF score range: {hybrid_recs['cf_score_norm'].min()} to {hybrid_recs['cf_score_norm'].max()}")
    
    # Calculate hybrid score using normalized scores
    hybrid_recs['hybrid_score'] = alpha * hybrid_recs['cb_score_norm'] + (1 - alpha) * hybrid_recs['cf_score_norm']
    
    # Sort by hybrid score
    hybrid_recs = hybrid_recs.sort_values('hybrid_score', ascending=False)
    
    # Filter out favorite openings
    hybrid_recs = hybrid_recs[~hybrid_recs.opening_name.isin(favorite_openings)]
    
    # Return detailed results for analysis
    result_df = hybrid_recs[['opening_name', 'hybrid_score', 'cb_score', 'cb_score_norm', 'cf_score', 'cf_score_norm']].head(top_n)
    
    # Simple verification
    print(f"Top hybrid recommendation scores - alpha={alpha}:")
    for idx, row in result_df.head(3).iterrows():
        print(f"  {row['opening_name']}: hybrid={row['hybrid_score']:.4f}, " +
              f"cb_norm={row['cb_score_norm']:.4f}, cf_norm={row['cf_score_norm']:.4f}")
    
    # Return only the needed columns for the final output
    return result_df[['opening_name', 'hybrid_score', 'cb_score', 'cf_score']]
    
# Function to display opening details
def display_opening_details(chess_data, opening_name, similarity_score=None, cf_score=None, hybrid_score=None):
    """
    Menampilkan detail opening untuk rekomendasi dengan informasi scoring yang tepat
    """
    opening_data = chess_data[chess_data.opening_name == opening_name]
    
    if len(opening_data) > 0:
        opening_data = opening_data.iloc[0]
        moves = opening_data['moves'].split(' ')[:opening_data['opening_ply']]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(opening_name)
            st.write(f"**Archetype:** {opening_data['opening_archetype']}")
            st.write(f"**Opening Moves:** {' '.join(moves)}")
            
            # Display scores with proper formatting
            st.write("### Recommendation Scores")
            scores_html = "<table style='width: 100%;'><tr><th>Score Type</th><th>Value</th></tr>"
            
            if similarity_score is not None:
                scores_html += f"<tr><td>Content-Based</td><td>{similarity_score:.4f}</td></tr>"
            
            if cf_score is not None:
                scores_html += f"<tr><td>Collaborative</td><td>{cf_score:.4f}</td></tr>"
            
            if hybrid_score is not None:
                scores_html += f"<tr><td><strong>Hybrid Score</strong></td><td><strong>{hybrid_score:.4f}</strong></td></tr>"
                
            scores_html += "</table>"
            st.markdown(scores_html, unsafe_allow_html=True)
                
        with col2:
            # Get win rates if available
            opening_games = chess_data[chess_data.opening_name == opening_name]
            if len(opening_games) > 0 and 'winner' in opening_games.columns:
                white_wins = sum(opening_games['winner'] == 'white')
                black_wins = sum(opening_games['winner'] == 'black')
                draws = sum(opening_games['winner'] == 'draw')
                total = len(opening_games)
                
                st.write("### Win Rates")
                st.write(f"White: {white_wins/total:.1%}")
                st.write(f"Black: {black_wins/total:.1%}")
                st.write(f"Draw: {draws/total:.1%}")
                
                # Create a pie chart for win rates
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(
                    [white_wins/total, black_wins/total, draws/total],
                    labels=['White', 'Black', 'Draw'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#f0f0f0', '#303030', '#909090']
                )
                ax.axis('equal')
                st.pyplot(fig)
    else:
        st.warning(f"No data found for opening: {opening_name}")

# Load data
chess_data = load_data()

if chess_data is not None:
    # Show dataset info
    with st.expander("Dataset Information"):
        st.write(f"Number of games: {len(chess_data.id.unique())}")
        st.write(f"Number of white players: {len(chess_data.white_id.unique())}")
        st.write(f"Number of black players: {len(chess_data.black_id.unique())}")
        st.write(f"Number of unique openings: {len(chess_data.opening_name.unique())}")
    
    # Try to load models
    content_based_model, collaborative_data, collaborative_model, hybrid_model = load_models()
    
    # Sidebar inputs
    st.sidebar.header("Your Chess Profile")
    
    # User rating
    user_rating = st.sidebar.slider("Your Rating", min_value=500, max_value=3000, value=1500, step=50)
    
    # Get unique openings for selection
    unique_openings = chess_data.opening_name.unique()
    
    # Select favorite openings
    st.sidebar.subheader("Select your 3 most played openings")
    opening_1 = st.sidebar.selectbox("Opening 1", options=unique_openings, index=0)
    opening_2 = st.sidebar.selectbox("Opening 2", options=unique_openings, index=1)
    opening_3 = st.sidebar.selectbox("Opening 3", options=unique_openings, index=2)
    
    favorite_openings = [opening_1, opening_2, opening_3]
    
    # Alpha parameter for hybrid recommendations
    alpha = st.sidebar.slider(
        "Content-Based vs Collaborative Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7,
        step=0.1,
        help="Higher values give more weight to content-based recommendations, lower values favor collaborative recommendations"
    )
    
    # Create tabs for different recommendation methods
    tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative Filtering", "Hybrid Filtering"])
    
    # Function to process inputs and generate recommendations
    def generate_recommendations():
        if content_based_model is None or collaborative_data is None or collaborative_model is None:
            st.error("Models not loaded. Please run the training notebook first.")
            return
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Compute content-based recommendations
        status_text.text("Computing content-based recommendations...")
        content_recs = get_content_based_recommendations(favorite_openings, content_based_model)
        progress_bar.progress(33)
        
        # Step 2: Compute collaborative filtering recommendations
        status_text.text("Computing collaborative filtering recommendations...")
        collab_recs = get_collaborative_recommendations(user_rating, collaborative_data, collaborative_model)
        progress_bar.progress(66)
        
        # Step 3: Compute hybrid recommendations
        status_text.text("Combining recommendations...")
        hybrid_recs = get_hybrid_recommendations(
            favorite_openings, 
            user_rating, 
            content_based_model, 
            collaborative_data, 
            collaborative_model,
            alpha=alpha
        )
        progress_bar.progress(100)
        status_text.text("Done!")
        
        # Display content-based recommendations
        with tab1:
            st.header("Content-Based Recommendations")
            st.write("Recommendations based on move similarity to your favorite openings")
            
            if len(content_recs) > 0:
                for i, (idx, row) in enumerate(content_recs.iterrows(), 1):
                    opening = row['opening_name']
                    score = row['similarity_score']
                    
                    with st.expander(f"{i}. {opening} (Similarity: {score:.2f})"):
                        display_opening_details(chess_data, opening, similarity_score=score)
            else:
                st.warning("No content-based recommendations found. Please try different openings.")
                
        # Display collaborative filtering recommendations
        with tab2:
            st.header("Collaborative Filtering Recommendations")
            st.write("Recommendations based on openings played by players with similar ratings")
            
            if len(collab_recs) > 0:
                for i, (idx, row) in enumerate(collab_recs.iterrows(), 1):
                    opening = row['opening_name']
                    score = row['score']
                    
                    with st.expander(f"{i}. {opening} (Score: {score:.2f})"):
                        display_opening_details(chess_data, opening, cf_score=score)
            else:
                st.warning("No collaborative filtering recommendations found. Please try a different rating.")
                
        # Display hybrid recommendations
        with tab3:
            st.header("Hybrid Recommendations")
            st.write("Combined recommendations from both content-based and collaborative filtering")
        
            if len(hybrid_recs) > 0:
                # First, display a summary table of all recommendations
                summary_df = hybrid_recs[['opening_name', 'hybrid_score', 'cb_score', 'cf_score']].copy()
                
                # Format scores for display
                summary_df['hybrid_score'] = summary_df['hybrid_score'].map('{:.4f}'.format)
                summary_df['cb_score'] = summary_df['cb_score'].map('{:.4f}'.format)
                summary_df['cf_score'] = summary_df['cf_score'].map('{:.4f}'.format)
                
                # Rename columns for better display
                summary_df.columns = ['Opening Name', 'Hybrid Score', 'Content-Based Score', 'Collaborative Score']
                
                # Display the summary table
                st.dataframe(summary_df, use_container_width=True)
                
                # Horizontal line
                st.markdown("---")
                
                # Then show detailed expandable sections
                for i, (idx, row) in enumerate(hybrid_recs.iterrows(), 1):
                    opening = row['opening_name']
                    hybrid_score = row['hybrid_score'] 
                    cb_score = row['cb_score'] if 'cb_score' in row else None
                    cf_score = row['cf_score'] if 'cf_score' in row else None
                    
                    with st.expander(f"{i}. {opening} (Hybrid Score: {hybrid_score:.4f})"):
                        display_opening_details(
                            chess_data, 
                            opening, 
                            similarity_score=cb_score,
                            cf_score=cf_score,
                            hybrid_score=hybrid_score
                        )
            else:
                st.warning("No hybrid recommendations found. Please try different settings.")
    
    # Generate recommendations button
    if st.sidebar.button("Get Recommendations"):
        generate_recommendations()
else:
    st.error("Failed to load dataset. Please check the file path.")

# Add information about the system
st.sidebar.markdown("---")
st.sidebar.info("""
    **About This System**
    
    This chess opening recommendation system uses:
    - Content-based filtering: Recommends openings based on move similarity
    - Collaborative filtering: Recommends openings based on preferences of players with similar ratings
    - Hybrid filtering: Combines both approaches for better recommendations
    
    To get started, select your rating and favorite openings, then click 'Get Recommendations'
""")

