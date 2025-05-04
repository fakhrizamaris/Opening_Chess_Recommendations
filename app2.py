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
        with open('models\content\models\content_based_model.pkl', 'rb') as f:
            content_based_model = pickle.load(f)
        
        # Load collaborative data
        with open('models\content\models\collaborative_data.pkl', 'rb') as f:
            collaborative_data = pickle.load(f)
        
        # Load collaborative model
        collaborative_model = {
            'model': tf.keras.models.load_model('models\content\models\collaborative_model.keras')
        }
        
        # Load additional collaborative model data
        with open('models\content\models\collaborative_model_data.pkl', 'rb') as f:
            collab_data = pickle.load(f)
            collaborative_model.update(collab_data)
        
        # Load hybrid model
        with open('models\content\models\hybrid_model.pkl', 'rb') as f:
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
    
    # Filter out favorite openings
    recommendations = recommendations[~recommendations.opening_name.isin(favorite_openings)]
    
    # Sort berdasarkan similarity score
    recommendations = recommendations.sort_values('similarity_score', ascending=False)
    
    return recommendations.head(top_n)
        
# Function to get collaborative filtering recommendations
def get_collaborative_recommendations(user_rating, collaborative_data, collaborative_model, top_n=5):
    """
    Mendapatkan rekomendasi opening berdasarkan rating pemain
    """
    # Cari pemain dengan rating yang mirip
    player_data = collaborative_data['player_data']
    rating_diff = abs(player_data['rating'] - user_rating)
    similar_players_idx = rating_diff.nsmallest(20).index
    similar_players = player_data.loc[similar_players_idx, 'player_id'].unique()
    
    # Jika tidak ada pemain yang mirip, return DataFrame kosong
    if len(similar_players) == 0:
        return pd.DataFrame(columns=['opening_name', 'score'])
    
    # Encode similar players
    player_encoder = collaborative_data['player_encoder']
    opening_encoder = collaborative_data['opening_encoder']
    
    # Filter similar players yang ada dalam encoder
    valid_similar_players = [p for p in similar_players if p in player_encoder.classes_]
    
    if len(valid_similar_players) == 0:
        return pd.DataFrame(columns=['opening_name', 'score'])
    
    # Encoded similar players
    encoded_similar_players = [player_encoder.transform([p])[0] for p in valid_similar_players]
    
    # Get all openings
    all_openings = opening_encoder.classes_
    encoded_all_openings = np.arange(len(all_openings))
    
    # Predict ratings for all openings for each similar player
    model = collaborative_model['model']
    
    # Initialize array for storing prediction scores
    all_predictions = np.zeros((len(valid_similar_players), len(all_openings)))
    
    # Make predictions
    for i, player_id in enumerate(encoded_similar_players):
        player_input = np.array([player_id] * len(all_openings))
        opening_input = encoded_all_openings
        
        predictions = model.predict(
            [player_input, opening_input],
            verbose=0
        )
        
        all_predictions[i, :] = predictions.flatten()
    
    # Average predictions across similar players
    avg_predictions = np.mean(all_predictions, axis=0)
    
    # Create recommendations DataFrame
    recommendations = pd.DataFrame({
        'opening_name': all_openings,
        'score': avg_predictions
    })
    
    # Sort by score
    recommendations = recommendations.sort_values('score', ascending=False)
    
    return recommendations.head(top_n)
    
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

# Function to get hybrid recommendations
def get_hybrid_recommendations(favorite_openings, user_rating, 
                              content_based_model, collaborative_data, collaborative_model,
                              alpha=0.7, top_n=5):
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
    
    # If either recommendation is empty, return the other
    if len(content_recs) == 0:
        return collab_recs.head(top_n)
    elif len(collab_recs) == 0:
        return content_recs.head(top_n)
    
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

