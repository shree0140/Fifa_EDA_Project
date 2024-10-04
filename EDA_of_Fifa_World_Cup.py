#!/usr/bin/env python
# coding: utf-8

# ### EDA
# 
# - Exploratory Data Analysis.

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[23]:


world_ranking= pd.read_csv("fifa_ranking_2022-10-06.csv")
world_cup= pd.read_csv("world_cup.csv")
matches = pd.read_csv("matches_1930_2022.csv")


# In[24]:


world_ranking #2018 , #2022


# In[25]:


world_cup


# In[26]:


matches.head()


# In[27]:


matches.info()


# In[28]:


matches['home_substitute_in_long'][0]


# In[29]:


print(matches.shape)
print(world_cup.shape)
print(world_ranking.shape)


# In[30]:


#Checking for Null values
matches.isnull().sum()/len(matches)*100


# In[31]:


world_ranking.isnull().sum()


# In[32]:


world_cup.isnull().sum()


# In[33]:


print(matches.duplicated().sum())
print(world_ranking.duplicated().sum())
print(world_cup.duplicated().sum())


# In[34]:


matches['home_team'].unique()


# In[35]:


#west germany or Germany DR - 1949 - 1990


# In[36]:


matches['home_team']= matches['home_team'].apply(lambda x: x.replace("Germany DR", " West Germany"))
matches['away_team']= matches['away_team'].apply(lambda x: x.replace("Germany DR", " West Germany"))


# # EDA:

# In[38]:


matches.info()


# In[39]:


# Total number of matches played till 2022.
print(f'Total Number of matches : {len(matches)}')

# Total goals Scored (penalty Goals).
print(f"Total goals scored: {sum(matches['home_score'] + matches['away_score'])}")

#Average Goals Scored Per match.
print(f"Average goals scored per match: {sum(matches['home_score'] + matches['away_score'])/len(matches)}")

#Total attendance Over time.
print(f"Total attendance Over time: {sum(matches['Attendance'])}")

#Average aattendance Over time.
print(f"Average aattendance Over time: {sum(matches['Attendance'])/len(matches)}")


# In[40]:


# Times a Particular country is winner , First runner up or  Second Runner up


# In[41]:


# WINNERS


# In[42]:


winner = world_cup['Champion'].value_counts().reset_index()


# In[56]:


winner.columns = ['Champion', 'count'] 


# In[57]:


fig = px.pie(winner, values='count', names= 'Champion', title= 'Number of times a Country won Fifa World Cup')

#customize
fig.update_traces(textinfo='label+value')
fig.show()





# In[58]:


#FIRST RUNNER UP AND SECOND RUNNER UP


# In[59]:


matches['Round'].unique()


# In[60]:


matches['home_penalty'].fillna(0, inplace=True)
matches['away_penalty'].fillna(0, inplace=True)


# In[61]:


matches['h_total']= matches['home_score'] + matches['home_penalty']
matches['a_total']= matches['away_score'] + matches['away_penalty']


# In[62]:


Runner = matches[matches['Round']=='Final']
Third = matches[matches['Round']=='Third-place match']


# In[64]:


#runner up
def get_runner_up(row):
    if(row['h_total']>row['a_total']):
        return row['away_team']
    else:
        return row['home_team']



#third place
def get_third_up(row):
    if(row['h_total']>row['a_total']):
        return row['home_team']
    else:
        return row['away_team']


# In[65]:


Runner['Runner_up'] = Runner.apply(get_runner_up, axis=1)
Third['Third_place'] = Third.apply(get_third_up, axis=1)


# In[66]:


Runner['Runner_up'].value_counts().index


# In[67]:


trace1= go.Bar(x=Runner['Runner_up'].value_counts().index, y=Runner['Runner_up'].value_counts(), name= 'First Runner Up')
trace2= go.Bar(x=Third['Third_place'].value_counts().index, y=Third['Third_place'].value_counts(), name= 'Second Runner Up')


# In[68]:


fig= make_subplots(rows=1, cols=2, subplot_titles=('First Runner Up', 'Second Runner Up'))

#Adding Traces
fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)

fig.show()




# ### CONCLUSIONS
# 
# - Brazil won the most  number of titles followed by Italy and Argentina.
# 
# - Argentina , Netherland and West Germany won Most number of Runner up title.
# 
# - Germany was Second runnner up for most of the times.
# 
# 

# In[69]:


# In which Year a Particular Team Participated.


# In[70]:


data= matches.groupby('Year').agg(home=('home_team', list), away=('away_team', list))
data


# In[71]:


data['teams']= data.apply(lambda x: list(set(x['home']+x['away'])), axis=1)

#1- concatinate
#2 - set ( duplicate X)
#3 - List


# In[72]:


data['#']= data['teams'].apply(len)


# In[73]:


data


# In[74]:


data_pivot = data['teams'].explode().reset_index().pivot(index='Year', columns='teams', values='Year').T


# In[75]:


data_pivot.fillna(0 , inplace=True)


# In[76]:


data_pivot= data_pivot.applymap(lambda x: 1 if x>0 else 0)


# In[77]:


plt.figure(figsize=(12,20))
sns.heatmap(data_pivot,linewidths= 0.5, linecolor='white')
plt.title("Team Participation Over Years")
plt.xlabel("Years")
plt.ylabel("Teams")
plt.show()


# ### OBSERVATIONS
# - Brazil is the only country to play all editions of the FIFA cup.
# 
# - Countries are Qatar , Canada , Wales are new to the tournament.
# 
# - Countires like Cuba, Israel , Kuwait , jamaica, Ukrain , Tongo etc. participated in only 1 edition of FIFA.

# In[78]:


#CHECKING FOR WHICH TEAM RANK INCREASED , DECREASED OR REMAIN SAME.


# In[79]:


Same_rank= world_ranking[world_ranking['rank']==world_ranking['previous_rank']]['team'].tolist()
Rank_decreased = world_ranking[world_ranking['rank']>world_ranking['previous_rank']]['team'].tolist()
Rank_increased = world_ranking[world_ranking['rank']<world_ranking['previous_rank']]['team'].tolist()


# In[80]:


Same_rank


# In[81]:


print(f"Same Rank : {len(Same_rank)}")
print(f"Rank Increased : {len(Rank_increased)}")
print(f"Rank Decreased : {len(Rank_decreased)}")


# In[82]:


max_length= max(len(Same_rank), len(Rank_increased), len(Rank_decreased))


# In[83]:


Same_rank += [None]*(max_length-len(Same_rank))
Rank_increased += [None]*(max_length-len(Rank_increased))
Rank_decreased += [None]*(max_length-len(Rank_decreased))


# In[84]:


Same_rank


# In[85]:


df= pd.DataFrame({
    "Same Rank": Same_rank,
    "Rank Increased": Rank_increased,
    "Rank Decreased": Rank_decreased
})


# In[86]:


df


# In[87]:


df.isnull().sum()


# In[88]:


df.fillna('', inplace=True)


# In[89]:


#Function to check rank increase or decrease.
def find_Rank(Country):
    result = []
    for column in df.columns:
        matches = df[column] == Country
        if matches.any():
            result.append(column)
    return result



# In[90]:


find_Rank("Belgium")


# # ADVANCE EDA:

# In[91]:


# PROBABILITY OF A TEAM WINNING KNOCKOUT MATCHES


# In[92]:


country ='Argentina'

#final
#semi final
#quarter final



# In[93]:


matches['Round'].unique()


# In[94]:


data = matches[((matches['home_team']==country) | (matches['away_team']==country))&
        ((matches['Round']=='Final') | (matches['Round']=='Semi-finals') | (matches['Round']=='Quarter-finals'))]


# In[95]:


df1= data['Round'].value_counts().reset_index()
df1.columns=['Round',"Total Matches"]


# In[97]:


#Functions to  determine the Winner of each game
def get_winner(row):
    if(row['h_total']>row['a_total']):
        return row['home_team']
    else:
        return row['away_team']


# In[98]:


data['Winner']= data.apply(get_winner, axis=1)


# In[99]:


df2= data[data['Winner']==country]['Round'].value_counts().reset_index()
df2.columns=['Round',"Total Matches"]


# In[100]:


pd.merge(df1, df2, on='Round', how='inner')


# In[101]:


def calculate_performance(country):
    # Filter matches based on the specified country and rounds
    data = matches[((matches['home_team'] == country) | (matches['away_team'] == country)) &
                   ((matches['Round'] == 'Final') | (matches['Round'] == 'Semi-finals') | (matches['Round'] == 'Quarter-finals'))]

    # Create a DataFrame with the count of matches per round
    df1 = pd.DataFrame(data['Round'].value_counts().reset_index())
    df1.columns = ['Round', 'Total Matches']

    # Function to determine the winner of each match
    def get_winner(row):
        if row['h_total'] > row['a_total']:
            return row['home_team']
        else:
            return row['away_team']

    # Apply the function to determine the winner for each match
    data['Winner'] = data.apply(get_winner, axis=1)

    # Create a DataFrame with the count of matches won by the specified country per round
    df2 = pd.DataFrame(data[data['Winner'] == country]['Round'].value_counts().reset_index())
    df2.columns = ['Round', 'Matches Won']

    # Merge the two DataFrames on the 'Round' column
    result = pd.merge(df1, df2, on='Round', how='inner')

    # Calculate the percentage of matches won
    result['Percentage'] = (result['Matches Won'] / result['Total Matches']) * 100

    return result




# In[102]:


calculate_performance('Argentina')


# ### CONCLUSIONS:
# 
# - Italy: Teams facing Italy in knockout rounds can expect a challenging match. Italy has a 75% probability of winning in the quarter-finals, an 86% probability of advancing from the semi-finals, and a 67% probability of securing victory in the finals.
# 
# - Spain - if it clears QF its wins title.
# - France - they have good record in QF  comparatively not that good record in Semi and Finals.
# 
# 
# - Sweden never reaches a final.
# 
# - if Argentina reaches semi final , it will surely win it.

# In[103]:


# CHECKING PLAYING STRATEGY OF  TEAMS IN KNOCKOUT V/S NON-KNOCKOUT MATCHES


# In[104]:


matches['home_goal'][0]


# In[105]:


# re -


# In[106]:


import re


# In[107]:


pattern = r'\b(\d+)\b'


# In[108]:


#Function to extract time from each row  for home_team

def extract_time(row):
  if pd.notna(row['home_goal']):
    goal_str= str(row['home_goal'])
    times=[]
    matches= re.findall(pattern, goal_str)
    times.extend(matches)
    return ', '.join(times)
  else:
    return ''


# In[109]:


matches['Home_Goals']= matches.apply(extract_time, axis=1)


# In[110]:


#Function to extract time from each row  for away team

def extract_time(row):
  if pd.notna(row['away_goal']):
    goal_str= str(row['away_goal'])
    times=[]
    matches= re.findall(pattern, goal_str)
    times.extend(matches)
    return ', '.join(times)
  else:
    return ''


# In[111]:


matches['Away_Goals']= matches.apply(extract_time, axis=1)


# In[112]:


country


# In[113]:


#CHECKING FOR KNOCKOUTS OR NOT

knockout_rounds= ['Final', 'Semi-finals', 'Quarter-finals']

matches['Knockout']= matches['Round'].apply(lambda x : 'Knockout' if x in knockout_rounds else 'Non-Knockout')


# In[114]:


country= 'Romania'


# In[115]:


home_k = matches[(matches['home_team']==country)&(matches['Knockout']== 'Knockout')]
away_k = matches[(matches['away_team']==country)&(matches['Knockout']== 'Knockout')]


home_nk = matches[(matches['home_team']==country)&(matches['Knockout']== 'Non-Knockout')]
away_nk = matches[(matches['away_team']==country)&(matches['Knockout']== 'Non-Knockout')]


# In[117]:


#FOR KNOCKOUT MATCHES
a=home_k['Home_Goals'].str.split(',').explode().reset_index()
b=away_k['Away_Goals'].str.split(',').explode().reset_index()

df1= pd.concat([a,b], axis=0, ignore_index=True)
df1.replace(r'^\s*$', 0, regex=True, inplace=True)
df1.fillna(0 ,inplace= True)


df1['Home_Goals']= df1['Home_Goals'].astype(int)
df1['Away_Goals']= df1['Away_Goals'].astype(int)

df1['Time']= df1['Home_Goals']+df1['Away_Goals']
df1= df1[df1['Time']!=0]

def determine(row):
      if(row['Time']<=45):
        return("First Half")
      elif(row['Time']>45 and row['Time']<=90):
        return("Second Half")
      else:
        return("Extra Time")

df1['time']= df1.apply(determine, axis=1)


# In[118]:


df1['time'].value_counts()/len(df1)*100


# In[119]:


# FOR NON-KNOCKOUT GAMES
a=home_nk['Home_Goals'].str.split(',').explode().reset_index()
b=away_nk['Away_Goals'].str.split(',').explode().reset_index()

df2= pd.concat([a,b], axis=0, ignore_index=True)
df2.replace(r'^\s*$', 0, regex=True, inplace=True)
df2.fillna(0 ,inplace= True)


df2['Home_Goals']= df2['Home_Goals'].astype(int)
df2['Away_Goals']= df2['Away_Goals'].astype(int)

df2['Time']= df2['Home_Goals']+df2['Away_Goals']
df2= df2[df2['Time']!=0]

def determine(row):
      if(row['Time']<=45):
        return("First Half")
      elif(row['Time']>45 and row['Time']<=90):
        return("Second Half")
      else:
        return("Extra Time")

df2['time']= df2.apply(determine, axis=1)


# In[120]:


df2['time'].value_counts()/len(df2)*100


# ### CONCLUSION:
# 
# - Generally teams play more agressive in 2nd half as compared to first half be it knockout or non- knockout tournaments.
# 
# - In knockout Tournaments , Mexico only scored in First half, either they are defending in second half and maintaining the lead.
# 
# - In knockout Tournaments countries Romaniya , Pero play with same agression in both the halves of the game.
# 
# - Spain have very different strategy they place with almost equal agression in non knockout matches , however in knowckout matches they follow different plan , on first half they are into passive state (not scoring much) , however in second half they retaliate very voilently. Infact goals scored by them in first half and extra time are equal in number.
# 
# 
# - Canada never reached a knowckout stage , and even in non knockout stages they never scored a goal in second half. This can be their  scope of improvement area , as they not scoring in the second half may be the reason they never made it to knockouts.

# In[122]:


# DOES SUBSTITUTION TURNS OUT TO BE FRUITFULL  OR NOT FOR THE TEAM.


#fruitfull --- substituted player score, Overall Score Improve


# In[123]:


matches['home_substitute_in_long'][0]



# in player ---- did he scored
# substituion time ---- does the score improved.


# we will use re library.


# In[125]:


matches_df= pd.read_csv('matches_1930_2022.csv')


# In[126]:


matches_df


# In[127]:


# Parse substituion column function
def parse_substitution_data(subs):
    if pd.isna(subs):
        return []
    pattern = r"(\d+)&rsquor;\|(.+?)\|for (.+?)(?=\||$)"
    matches = re.findall(pattern, subs)   #return a tuple , of time , player in ,player out.
    return [{'minute': int(match[0]), 'player_in': match[1], 'player_out': match[2]} for match in matches]


# In[128]:


matches_df['home_substitute_in_long'].apply(parse_substitution_data)[0]


# In[129]:


matches_df['home_substitutions'] = matches_df['home_substitute_in_long'].apply(parse_substitution_data)
matches_df['away_substitutions'] = matches_df['away_substitute_in_long'].apply(parse_substitution_data)


# In[130]:


# did this substitute scored a goal or improved performance


# In[143]:


# Parsing Goal Data
def parse_goal_data(goals):
    if pd.isna(goals):
        return []
        
    goal_list = goals.split('|')
    parsed_goals = []
    
    for goal in goal_list:
        parts = goal.split('·')  # middle dot or interpunct  (alt+0183)
        if len(parts) == 2:
            try:
                minute = int(parts[1].strip())
                scorer = parts[0].strip()
                parsed_goals.append({'minute': minute, 'scorer': scorer})
            except ValueError:
                pass
    
    return parsed_goals



# In[144]:


matches_df['home_goal_details']=matches['home_goal'].apply(parse_goal_data)
matches_df['away_goal_details']=matches['away_goal'].apply(parse_goal_data)


# In[145]:


# i know which player was substituted
# which player scored and at which time.


# In[146]:


def filter_matches_for_team(df, team_name):
  team_matches= df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].copy()
  return team_matches


# In[147]:


team_name= 'Germany'
team_matches_df= filter_matches_for_team(matches_df, team_name)


# In[148]:


# if substituted Player scored a goal
# we will create two functions.

#subs- list {substituions}
#goals- list {goals}

def check_substitute_goals(subs, goals):
    if not subs:
        return
    for sub in subs:
        if any(goal['scorer'] == sub['player_in'] for goal in goals):
            return True
    return False



def check_team_substitute_goals(row, team_name):
    if row['home_team'] == team_name:
        return check_substitute_goals(row['home_substitutions'], row['home_goal_details'])
    elif row['away_team'] == team_name:
        return check_substitute_goals(row['away_substitutions'], row['away_goal_details'])
    return False


# In[149]:


# Determine if substitution improved the score
def check_substitution_impact(row, team_name):
    if row['home_team'] == team_name:
        subs = row['home_substitutions']
        goals_before = [goal for goal in row['home_goal_details'] if subs and goal['minute'] <= subs[0]['minute']] if subs else []
        goals_after = [goal for goal in row['home_goal_details'] if subs and goal['minute'] > subs[0]['minute']] if subs else []
    elif row['away_team'] == team_name:
        subs = row['away_substitutions']
        goals_before = [goal for goal in row['away_goal_details'] if subs and goal['minute'] <= subs[0]['minute']] if subs else []
        goals_after = [goal for goal in row['away_goal_details'] if subs and goal['minute'] > subs[0]['minute']] if subs else []
    else:
        return False


    return len(goals_after) > len(goals_before)


# In[150]:


team_matches_df['substitution_impact'] = team_matches_df.apply(lambda row: check_substitution_impact(row, team_name), axis=1)
team_matches_df['substitute_goals'] = team_matches_df.apply(lambda row: check_team_substitute_goals(row, team_name), axis=1)


# In[151]:


match_summaries = []

for index, row in team_matches_df.iterrows():
    match_summary = {
        'Match': f"{row['home_team']} vs {row['away_team']}",
        'Date': row['Date'],
        'Substitute Scored': row['substitute_goals'],
        'Substitution Impact': row['substitution_impact']
    }
    match_summaries.append(match_summary)


# In[152]:


summary_df = pd.DataFrame(match_summaries)


# In[153]:


print(summary_df['Substitute Scored'].value_counts())
print('***')
print(summary_df['Substitution Impact'].value_counts())


# ### CONCLUSIONS:
# 
# - No Team substituted player scored goal in FIFA.
# 
# - Teams like France , Argentina , Portugal , Australia , Germany , Spain , Netherland have very  high or positive impact of subsutituion on overall score of the team.
# 
# - Positive impact of the substituion signifies that these teams have a strong management that act accourdingly , as these teams are also the top teams of FIFA, these effective substution may be  one of the reason why these teams are most favourite teams of the world.
# 
# - A team can rely on substituted player for overall improvement of score but not for the goals to be scored by substituted player.

# In[154]:


#Team expected Performance and Actual Performance


# In[155]:


#Calculate expected goals  and total goals for  each team

home_team_stats= matches.groupby(['home_team','Year']).agg({
    'home_score':'sum',
    'home_xg' :'sum'
}).rename(columns={'home_score':'Total_goals'})

away_team_stats= matches.groupby(['away_team','Year']).agg({
    'away_score':'sum',
    'away_xg' :'sum'
}).rename(columns={'away_score':'Total_goals'})


# In[156]:


home_team_stats= home_team_stats.reset_index()
away_team_stats= away_team_stats.reset_index()


home_team_stats= home_team_stats[(home_team_stats['Year']==2018 )|(home_team_stats['Year']==2022)]
away_team_stats= away_team_stats[(away_team_stats['Year']==2018 )|(away_team_stats['Year']==2022)]


# In[157]:


home_team_stats.rename(columns= {'home_team':'team'}, inplace=True)
away_team_stats.rename(columns= {'away_team':'team'}, inplace=True)


# In[158]:


merged= pd.merge(home_team_stats, away_team_stats, on=['team','Year'], how='inner')


# In[159]:


merged['Total_goals']= merged['Total_goals_x']+merged['Total_goals_y']
merged['Total_xg']= merged['home_xg']+merged['away_xg']
merged['Deviation']=merged['Total_xg']-merged['Total_goals']


# In[160]:


merged[merged['Year']==2018]


# ### CONCLUSION.
# 
# - IN 2018 , Argentina Failed to take home advantages and underperformed in home games , However they over performed in non home grounds.
# 
# - In 2018 Russia Was the most attacking team , as they scored 60% more goals  then they were expected to score , this may be due to home condition advantage as in 2018 FIFA world cup was in Russia.
# 
# - Russia  was also the Underdog Team in 2018 , as it performed much more than expected.
# 
# - Netherland was the underdog team of  2022, it scored  52% more goals than expected.
# 
# - Among top teams Portugal scored  40% more than expected goals.
# 
# - Most Overrated team of 2018 and 2022 was Brazil  they scored
#  32%  and 40% less goals than expected in  2018 and 2022 respectively.

# In[ ]:




