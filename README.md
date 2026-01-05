# FM2020 Player Similarity Finder

An interactive data engineering and machine learning tool designed to identify matching player profiles within a 94k record dataset. By treating player attributes as 47-dimensional vectors, the system calculates similarity scores to find alternatives for specific players who can play the same way or be retrained.

## ✨ Key Features
- **Dynamic Search:** Real-time similarity indexing for over 94,000 player records
- **Multi-Factor Filtering:** Fine-tune results by Age, Market Value, and Weekly Wage to find realistic transfer targets
- **Visual Analytics:** Interactive **Plotly Radar Charts** for attribute comparison
- **Optimized UI:** Built with **Streamlit Fragments** to ensure smooth performance during heavy data visualization tasks

## 🛠️ Data Engineering & Pipeline
- **ETL Process:** Developed a robust cleaning pipeline to handle non-standard HTML table exports, including currency normalization (£M/£K to Integer) and string parsing for physical metrics
- **Feature Engineering:** Implemented a dual-track attribute system, separating Goalkeeper and Outfield metrics to ensure domain-specific model accuracy
- **Normalization:** Utilized `MinMaxScaler` to bring all 47 technical attributes into a uniform range, preventing bias in the distance calculation

## 🤖 Machine Learning Logic
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Distance Metric:** **Cosine Similarity**. Cosine Similarity focuses on the shape of the attribute spread rather than raw magnitude, making it ideal for finding players with similar styles

## 💻 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn, NumPy
- **Interface:** Streamlit
- **Visuals:** Plotly

## 🚀 How to Run Locally
To run this project on your machine, follow these steps:

# 1. Clone the repository
git clone [https://github.com/Chazz236/FootballManagerSimilarityTool.git](https://github.com/Chazz236/FootballManagerSimilarityTool.git)

cd FootballManagerSimilarityTool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run similarity_tool.py
