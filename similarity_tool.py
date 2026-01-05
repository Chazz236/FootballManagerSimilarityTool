import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import streamlit as st #https://docs.streamlit.io/develop/api-reference
import plotly.graph_objects as go #https://plotly.com/python/radar-chart/

#input and output files
html_file = 'fm2020_players_2019.html'
output_csv = 'fm2020_players_2019.csv'

#defining all the attributes
attr_cols = ['1v1', 'Acc', 'Aer', 'Agg', 'Agi', 'Ant', 'Bal', 'Bra', 'Cmd', 'Cmp', 'Cnt', 'Com', 'Cor', 'Cro', 'Dec', 'Det', 'Dri', 'Ecc', 'Fin', 'Fir', 'Fla', 'Fre', 'Han', 'Hea', 'Jum', 'Kic', 'L Th', 'Ldr', 'Lon', 'Mar', 'Nat Fit', 'OtB', 'Pac', 'Pas', 'Pen', 'Pos', 'Pun', 'Ref', 'Sta', 'Str', 'Tck', 'Tea', 'Tec', 'Thr', 'TRO', 'Vis', 'Wor']

#separate gk attributes from outfield attributes
gk_attr = ['Aer', 'Cmd', 'Com', 'Ecc', 'Han', 'Kic', '1v1', 'Pun', 'Ref', 'TRO', 'Thr']
outfield_attr = [col for col in attr_cols if col not in gk_attr]

#add some necessary gk attributes, including sweeper keeper attributes as well
gk_attr = gk_attr + ['Fir', 'Pas', 'Ant', 'Cmp', 'Cnt', 'Dec', 'Pos', 'Vis', 'Acc', 'Agi']

def value_cleaner(value):
    #remove currency symbol
    value = value.replace('£', '')
    
    #if value is in millions or thousands, adjust accordingly
    if (value.find('M') != -1):
        value = float(value.replace('M', '')) * 1000000
    elif (value.find('K') != -1):
        value = float(value.replace('K', '')) * 1000
    
    return int(value)
    
def wage_cleaner(wage):
    #if player has no wage, put 0
    if (wage == '-' or pd.isna(wage)):
        return 0
    
    #remove string characters to return a integer value
    wage = wage.replace('£', '')
    wage = wage.replace(' p/w', '')
    wage = wage.replace(',', '')
    
    return int(wage)

def clean_and_process(html_file, output_csv):
    try:
        #convert html table to dataframe
        df = pd.read_html(html_file, flavor='html5lib')[0]

        #drop first nat column, which is nationality, rename other nat to nat fit
        cols = list(df.columns)
        cols[6] = 'Nationality'
        cols[31] = 'Nat Fit'
        df.columns = cols
        df = df.drop(columns = ['Nationality'])

        #remove columns that aren't necessary
        df = df.drop(columns = ['Inf', 'Yth Apps', 'Caps', 'Team', 'Goals', 'Based', 'NoB', 'Yth Gls', 'EU National'])

        #replace empty cells with - (only club has empty cells)
        df = df.replace('', '-') 

        #remove nationality from name
        df['Name'] = df['Name'].str.split(' - ').str[0]

        #convert age to int
        df['Age'] = df['Age'].astype(int)

        #remove cm from height
        df['Height'] = df['Height'].str.replace(' cm', '').astype(int)

        #remove kg from weight
        df['Weight'] = df['Weight'].str.replace(' kg', '').astype(int)

        #clean up value
        df['Value'] = df['Value'].apply(value_cleaner)

        #clean up wage
        df['Wage'] = df['Wage'].apply(wage_cleaner)

        #replace nan with a -
        df['Club'] = df['Club'].fillna('-')
       
        #make int - age, height, weight, caps, attributes
        df[attr_cols] = df[attr_cols].astype(int)

        #order cols:
        ordered_cols = ['Name', 'Age', 'Height', 'Weight', 'Personality', 'Position', 'Club', 'Division', 'Value', 'Wage', 'Expires', 'Min Fee Rls']
        df = df[ordered_cols + attr_cols]

        #create csv file
        df.to_csv(output_csv, encoding='utf-8')
        
        # print(df.info())
        
        return df

    except Exception as e:
        #print error
        print(f'Error, it was: {e}')

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def find_players(index, max_age = 100, max_value = 200000000,max_wage = 1000000):
    #check if player is gk or outfield, use according finder
    if df.iloc[index]['Position'] == 'GK':
        distances, indices = gk_finder.kneighbors(gk_df_scaled[index].reshape(1, -1), n_neighbors=500)
    else:
        distances, indices = outfield_finder.kneighbors(outfield_df_scaled[index].reshape(1, -1), n_neighbors=500)
    
    #add similarity column to see how close compares are to player
    similars = df.iloc[indices[0]].copy()
    similars['Similarity'] = 1 - distances[0]

    #mask to apply filters and exclude selected player
    mask = ((similars['Age'] <= max_age) & (similars['Value'] <= max_value) & (similars['Wage'] <= max_wage) & (similars.index != index))
    filtered = similars[mask]

    #return 10 most similar players
    return filtered.head(10)[['Name', 'Age', 'Position', 'Club', 'Value', 'Wage', 'Expires', 'Min Fee Rls', 'Personality', 'Similarity']]

#to create the radar chart
def radars(player_id, compare_id, attributes):
    #get player data for selected player and compare player
    player_data = df.iloc[player_id]
    compare_data = df.iloc[compare_id]

    player_name = player_data['Name']
    compare_name = compare_data['Name']

    figure = go.Figure()

    #add selected player trace to the graph
    figure.add_trace(go.Scatterpolar(
        r=[player_data[a] for a in attributes],
        theta=attributes,
        fill='toself',
        name=player_name
    ))
    
    #add compare placer trace to the graph
    figure.add_trace(go.Scatterpolar(
        r=[compare_data[a] for a in attributes],
        theta=attributes,
        fill='toself',
        name=compare_name
    ))

    #set up graph layout
    figure.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,20])),
        showlegend=True,
        title=f'Comparison: {player_name} vs {compare_name}',
    )

    return figure

#use fragment to update chart only
@st.fragment
def display(results, selected_index):
    st.subheader(f'Players similar to {df.iloc[selected_index]['Name']}')
    st.write('Click to see attribute comparison')

    #table with selection
    event = st.dataframe(data=results.style.format({'Similarity': '{:.2%}', 'Value': '£{:,}','Wage': '£{:,}'}), hide_index=True, on_select='rerun', selection_mode='single-row')
    
    #if user has selected a row, display chart, otherwise tell them to
    if len(event.selection.rows) > 0:
        compare_id = results.index[event.selection.rows[0]]
    
        #display overlapping radar chart to show how similar the attribute spread is
        st.subheader('Attribute Comparison')
        fig = radars(selected_index, compare_id, attr_cols)
        st.plotly_chart(fig, width='stretch', height=600)
        show_cols = ['Name', 'Age', 'Height', 'Weight', 'Personality', 'Position', 'Club', 'Division', 'Value', 'Wage', 'Expires', 'Min Fee Rls'] + attr_cols
        st.dataframe(data=df.iloc[[selected_index, compare_id]][show_cols].style.format({'Similarity': '{:.2%}', 'Value': '£{:,}','Wage': '£{:,}'}), hide_index=True)
    else:
        st.subheader('Attribute Comparison')
        st.write('Select a player from the table too see the attribute spread')

#cleaning, preprocessing, feature selection
# df = clean_and_process(html_file, output_csv)

#read csv if clean_and_process is done
df = load_data(output_csv)

#set up dfs for outfield and gk
outfield_df = df[outfield_attr]
gk_df = df[gk_attr]

#scale dfs with minmax so that all relevant attributes are weighed evenly
outfield_df_scaled = MinMaxScaler().fit_transform(outfield_df)
gk_df_scaled = MinMaxScaler().fit_transform(gk_df)

# using cosine similarity to find players with similar attribute spread
outfield_finder = NearestNeighbors(metric='cosine')
gk_finder = NearestNeighbors(metric='cosine')

#fit nearest neighbours estimator using scaled dfs
outfield_finder.fit(outfield_df_scaled)
gk_finder.fit(gk_df_scaled)

#streamlit part

#make the page wide to fit table
st.set_page_config(layout='wide')

#writing streamlit title and description
st.title('FM2020 Player Similarity Finder')
st.write('Find similar players based on their attribute spread')

#create unique identifier for searching in streamlit
df['Search'] = df.index.astype(str) + ' - ' + df['Name'] + ', ' + df['Club']

#setting up sidebar for streamlit
st.sidebar.header('Search Settings')
player_list = df['Search'].tolist()
player_map = dict(zip(df['Search'], df.index))

#select the player to find similar players
selected_player_name = st.sidebar.selectbox('Select a Player', options=list(player_map.keys()), index=0)
selected_index = player_map[selected_player_name]

#filters for finding similar players, wage and value are formated
age = st.sidebar.slider('Max Age', min_value=16, max_value=45, value=45)
wage = st.sidebar.select_slider('Max Weekly Wage (£)', options=[0, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000], value=1000000, format_func=lambda x: f'£{x:,}')
value = st.sidebar.select_slider('Max Value (£)', options=[0, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 25000000, 50000000, 100000000, 200000000], value=200000000, format_func=lambda x: f'£{x:,}')

#button to start search
search = st.sidebar.button('Find Similar Players')

if search:
    #spinner for loading
    with st.spinner('Calculating...'):
        #call function to get similar players
        results = find_players(selected_index, age, value, wage)

        #if similar players exist with filters, then show at most 10 similar players, otherwise display warning
        if not results.empty:
            display(results, selected_index)
        else:
            st.warning('No players found matching filters')