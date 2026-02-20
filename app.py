import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Handle KeyboardInterrupt gracefully ----------
try:
    # Page configuration (MUST be the first Streamlit command)
    st.set_page_config(
        page_title="Tourism Experience Analytics",
        page_icon="✈️",
        layout="wide"
    )

    # Title
    st.title("✈️ Tourism Experience Analytics")
    st.markdown("""
    This application provides **personalized attraction recommendations** and **visit mode predictions** 
    based on user profiles and historical data.
    """)

    # ---------- Data loading function with proper error handling ----------
    @st.cache_data
    def load_data():
        # 🔧 PUT YOUR ACTUAL CSV PATH HERE (or leave as is to use sample data)
        file_path = r'D:\Tourism\master_tourism_data.csv'   # <-- CHANGE THIS TO YOUR FILE
        
        # Check if file exists
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                st.success("✅ Original data loaded successfully!")
                return df
            except PermissionError:
                st.warning("⚠️ File is open in another program (e.g., Excel). Using sample data.")
            except Exception as e:
                st.warning(f"⚠️ Problem reading file: {e}. Using sample data.")
        else:
            st.info("ℹ️ CSV file not found. Generating sample data...")
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'UserId': np.random.randint(1, 101, n_samples),
            'AttractionId': np.random.randint(1, 51, n_samples),
            'Rating': np.random.randint(1, 6, n_samples),
            'VisitYear': np.random.randint(2018, 2023, n_samples),
            'VisitMonth': np.random.randint(1, 13, n_samples),
            'VisitMode': np.random.choice(['Business', 'Family', 'Couples', 'Friends', 'Solo'], n_samples),
            'Continent': np.random.choice(['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Australia'], n_samples),
            'Country': np.random.choice(['USA', 'India', 'UK', 'France', 'Japan', 'Australia'], n_samples),
            'Attraction': [f'Attraction {i}' for i in np.random.randint(1, 51, n_samples)]
        })
        return df

    # ---------- Model loading function ----------
    @st.cache_resource
    def load_models():
        models = {}
        if os.path.exists('visitmode_classifier_model.pkl') and os.path.exists('label_encoder_target.pkl'):
            models['clf'] = joblib.load('visitmode_classifier_model.pkl')
            models['target_encoder'] = joblib.load('label_encoder_target.pkl')
        else:
            models['clf'] = None
            models['target_encoder'] = None
        
        if os.path.exists('label_encoder_visitmode.pkl'):
            models['le_visitmode'] = joblib.load('label_encoder_visitmode.pkl')
        else:
            models['le_visitmode'] = None
            
        if os.path.exists('label_encoder_continent.pkl'):
            models['le_continent'] = joblib.load('label_encoder_continent.pkl')
        else:
            models['le_continent'] = None
            
        if os.path.exists('label_encoder_country.pkl'):
            models['le_country'] = joblib.load('label_encoder_country.pkl')
        else:
            models['le_country'] = None
        
        return models

    # ---------- Recommendation matrices loading ----------
    @st.cache_resource
    def load_recommender(df):
        if os.path.exists('user_item_matrix.pkl') and os.path.exists('user_sim_matrix.pkl'):
            with open('user_item_matrix.pkl', 'rb') as f:
                user_item = pickle.load(f)
            with open('user_sim_matrix.pkl', 'rb') as f:
                user_sim = pickle.load(f)
            st.info("📁 Loaded saved recommendation matrices.")
        else:
            st.warning("⚙️ Recommendation matrices not found, computing now...")
            user_item = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
            user_sim = pd.DataFrame(
                cosine_similarity(user_item),
                index=user_item.index,
                columns=user_item.index
            )
            # Save for next time
            with open('user_item_matrix.pkl', 'wb') as f:
                pickle.dump(user_item, f)
            with open('user_sim_matrix.pkl', 'wb') as f:
                pickle.dump(user_sim, f)
            st.success("✅ Recommendation matrices computed and saved!")
        return user_item, user_sim

    # ---------- Load everything ----------
    with st.spinner("Loading data and models..."):
        df = load_data()
        models = load_models()
        user_item_matrix, user_sim_matrix = load_recommender(df)

    # Attraction mapping (for global use)
    if 'Attraction' in df.columns and 'AttractionId' in df.columns:
        attraction_map = df[['AttractionId', 'Attraction']].drop_duplicates(subset='AttractionId').set_index('AttractionId')['Attraction']
    else:
        attraction_map = None

    st.success("✅ Everything loaded successfully!")

    # ---------- Sidebar ----------
    st.sidebar.header("🔍 User Input")
    user_list = sorted(df['UserId'].unique())
    selected_user = st.sidebar.selectbox("Select User ID", user_list)

    st.sidebar.subheader("Custom Visit Details")
    custom_year = st.sidebar.slider("Visit Year", 2018, 2023, 2022)
    custom_month = st.sidebar.slider("Visit Month", 1, 12, 6)
    custom_continent = st.sidebar.selectbox("Continent", df['Continent'].unique())
    custom_country = st.sidebar.selectbox("Country", df['Country'].unique())

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 User Profile", "🔮 Predict Visit Mode", "🎯 Recommendations", "📈 Analytics"])

    # ---------- Tab1: User Profile ----------
    with tab1:
        st.header(f"User Profile: {selected_user}")
        user_data = df[df['UserId'] == selected_user]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Visits", len(user_data))
        with col2:
            st.metric("Average Rating", f"{user_data['Rating'].mean():.2f}")
        with col3:
            st.metric("Most Common Mode", user_data['VisitMode'].mode()[0] if not user_data.empty else "N/A")
        
        st.subheader("Visit History")
        st.dataframe(user_data[['Attraction', 'VisitYear', 'VisitMonth', 'VisitMode', 'Rating']].head(10))
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(user_data['Rating'], bins=5, edgecolor='black', alpha=0.7)
        ax[0].set_title(f"User {selected_user} - Rating Distribution")
        ax[0].set_xlabel("Rating")
        ax[0].set_ylabel("Frequency")
        
        user_data['VisitMode'].value_counts().plot(kind='bar', ax=ax[1], color='coral')
        ax[1].set_title("Visit Modes")
        ax[1].set_xlabel("Mode")
        ax[1].set_ylabel("Count")
        st.pyplot(fig)

    # ---------- Tab2: Predict Visit Mode ----------
    with tab2:
        st.header("🔮 Predict Visit Mode")
        st.markdown("Predict the most likely visit mode based on user profile.")
        
        if st.button("Predict for Selected User"):
            if models['clf'] is not None:
                user_stats = {
                    'Rating': user_data['Rating'].mean(),
                    'TransactionId': len(user_data)
                }
                user_mode = user_data['VisitMode'].mode()[0] if not user_data.empty else 'Solo'
                user_continent = user_data['Continent'].mode()[0] if not user_data.empty else 'Asia'
                user_country = user_data['Country'].mode()[0] if not user_data.empty else 'India'
                
                visitmode_enc = models['le_visitmode'].transform([user_mode])[0] if models['le_visitmode'] else 0
                continent_enc = models['le_continent'].transform([user_continent])[0] if models['le_continent'] else 0
                country_enc = models['le_country'].transform([user_country])[0] if models['le_country'] else 0
                
                attraction_avg = df['Rating'].mean()
                attraction_count = len(df['AttractionId'].unique())
                
                features = [[custom_year, custom_month, visitmode_enc, continent_enc, country_enc,
                            user_stats.get('Rating', 3.0), user_stats.get('TransactionId', 1),
                            attraction_avg, attraction_count]]
                
                pred_enc = models['clf'].predict(features)[0]
                pred_mode = models['target_encoder'].inverse_transform([pred_enc])[0]
                st.success(f"✅ Predicted Visit Mode: **{pred_mode}**")
            else:
                st.warning("Classification model not trained. Using sample prediction.")
                st.info(f"Sample prediction: **{np.random.choice(['Business', 'Family', 'Couples', 'Friends', 'Solo'])}**")

    # ---------- Tab3: Recommendations ----------
    with tab3:
        st.header("🎯 Personalized Attraction Recommendations")
        
        def get_recommendations(user_id, user_item_df, user_sim_df, df, top_n=5):
            if user_id not in user_item_df.index:
                return []
            
            user_rated = user_item_df.loc[user_id]
            rated_attractions = user_rated[user_rated > 0].index.tolist()
            similar_users = user_sim_df.loc[user_id].sort_values(ascending=False).iloc[1:11].index
            
            all_attractions = user_item_df.columns
            unrated = [a for a in all_attractions if a not in rated_attractions]
            
            scores = []
            for att in unrated:
                weighted_sum = 0
                sim_sum = 0
                for sim_user in similar_users:
                    rating = user_item_df.loc[sim_user, att]
                    if rating > 0:
                        sim = user_sim_df.loc[user_id, sim_user]
                        weighted_sum += sim * rating
                        sim_sum += sim
                pred_score = weighted_sum / sim_sum if sim_sum > 0 else 0
                scores.append((att, pred_score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_n]
        
        recommendations = get_recommendations(selected_user, user_item_matrix, user_sim_matrix, df)
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations, columns=['AttractionId', 'Predicted Score'])
            
            if attraction_map is not None:
                rec_df['Attraction'] = rec_df['AttractionId'].map(attraction_map)
            else:
                rec_df['Attraction'] = rec_df['AttractionId'].apply(lambda x: f"Attraction {x}")
            
            st.subheader("Top 5 Recommended Attractions")
            st.dataframe(rec_df[['Attraction', 'Predicted Score']].style.format({'Predicted Score': '{:.3f}'}))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.viridis(np.linspace(0, 1, len(rec_df)))
            ax.barh(rec_df['Attraction'], rec_df['Predicted Score'], color=colors)
            ax.set_xlabel("Predicted Score")
            ax.set_title("Recommended Attractions")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.warning("No recommendations available for this user.")

    # ---------- Tab4: Analytics ----------
    with tab4:
        st.header("📈 Tourism Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots()
            df['Rating'].hist(bins=5, edgecolor='black', ax=ax)
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Visit Modes Distribution")
            fig, ax = plt.subplots()
            df['VisitMode'].value_counts().plot(kind='bar', ax=ax, color='coral')
            ax.set_xlabel("Mode")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        
        st.subheader("Top Attractions by Visits")
        top_att = df['AttractionId'].value_counts().head(10).reset_index()
        top_att.columns = ['AttractionId', 'Visit Count']
        if attraction_map is not None:
            top_att['Attraction'] = top_att['AttractionId'].map(attraction_map)
        else:
            top_att['Attraction'] = top_att['AttractionId'].apply(lambda x: f"Attraction {x}")
        st.dataframe(top_att[['Attraction', 'Visit Count']])
        
        st.subheader("Yearly Visit Trend")
        yearly = df.groupby('VisitYear').size().reset_index(name='Visits')
        fig, ax = plt.subplots()
        ax.plot(yearly['VisitYear'], yearly['Visits'], marker='o', linewidth=2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Visits")
        ax.grid(True)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("© 2024 Tourism Experience Analytics | Built with Streamlit")

except KeyboardInterrupt:
    st.warning("⚠️ Program manually stopped by user.")
    # If you need to perform cleanup, add it here.
    # The script will exit after this block.