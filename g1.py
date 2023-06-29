"""
CSE 163 Final Project
Group P059:
This file implements three data visualizations
using pandas, plotly, and ML. Sleepless Nights for
Insomnia through comparing the sleep duration by
age and gender. Secondly, through heart rate and
Lastly, through caffine alcohol. Lastly, the final
function implements ML in a line chart showing the trends
of heart rate and sleep quality over time.
"""


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import math


def bar_chart_sleep_data(data: pd.DataFrame) -> None:
    """
    This function takes in a sleep data dataset and produces a grouped
    bar chart. It compares the sleep duration by age and gender.
    The x-axis represents the age groups, and y-axis represents
    the sleep duration.
    """
    # Filters data for age group 40-59
    filtered_data = data[(data['Age'] >= 40) & (data['Age'] <= 59)]

    # Groups the filtered data by 'Age' and 'Gender' columns and
    # calculate the mean of 'Sleep duration'
    grouped_data = (
        filtered_data.groupby(['Age', 'Gender'])['Sleep duration']
        .mean()
        .unstack()
    )
    grouped_data = grouped_data.dropna(how='any')
    grouped_data = grouped_data.reset_index()

    # Plotly Express
    fig = px.bar(
        grouped_data,
        x='Age',
        y=['Female', 'Male'],
        barmode='group',
        labels={'value': 'Sleep Duration (Hours)'},
        title='Sleep Duration by Age and Gender',
        color_discrete_sequence=['#ADD8E6', '#FFB6C1']
    )

    # Layout
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': grouped_data['Age']},
        legend={'title': 'Gender'},
        title={'text': 'Sleep Duration by Age and Gender', 'x': 0.5,
               'y': 0.95, 'font': {'size': 24}},
        xaxis_title={'text': 'Age', 'font': {'size': 24}},
        yaxis_title={'text': 'Sleep Duration (Hours)', 'font': {'size': 24}}
    )
    fig.update_traces(width=0.45)

    # HTML FILE
    fig.write_html('bar_chart_sleep_data.html')


def sleep_stresslevel_scatterplot(data: pd.DataFrame) -> None:
    """
    Takes in a dataset and produce a scatterplot
    that indicates the relationship between
    sleep quality,length,and stress level
    (x-axis is sleep quality, y-axis is sleep duration,
    and color represents stress level,
    title "Quality & Length sleep with
    stress level of insomnia people age 40-59")
    Also, measures the correlation between those variables.
    """
    # Filter the data to include ages from 40 to 59
    senior_filter = (data['Age'] >= 40) & (data['Age'] <= 59)

    # Filter the data to include rows where "Sleep Disorder" is "Insomnia"
    insomnia_filter = data['Sleep Disorder'] == 'Insomnia'
    filter_data = data.loc[senior_filter & insomnia_filter]

    # Create the scatterplot using Plotly
    fig = px.scatter(
        filter_data,
        x="Quality of Sleep",
        y="Sleep Duration",
        color="Stress Level",
        title=("Quality & Length sleep with stress level" +
               " of insomnia people age 40-59"),
        labels={"Quality of Sleep": "Sleep Quality",
                "Sleep Duration": "Sleep Duration"},
        hover_data=['Age', 'Gender']
    )
    fig.update_traces(marker=dict(size=15, opacity=0.5),
                      selector=dict(mode='markers'))
    fig.write_html("scatterplot.html")

    # Calculate correlation "Quality of Sleep" & "Sleep Duration"
    correlation = filter_data["Quality of Sleep"].corr(
                  filter_data["Sleep Duration"])
    print("Correlation between Quality" +
          " of Sleep and Sleep Duration:")
    print(correlation)
    print()

    # correlation table between 3 variables:
    correlation2 = filter_data[['Quality of Sleep', 'Sleep Duration',
                                'Stress Level']].corr()
    print("Correlation matrix:")
    print(correlation2)
    print()

    # correlation for dependent level of "Stress Level"
    cor = filter_data.corr()
    # Independent variables
    x = "Quality of Sleep"
    y = "Sleep Duration"
    # Dependent variable
    z = 'Stress Level'
    # Pairings
    xz = cor.loc[x, z]
    yz = cor.loc[y, z]
    xy = cor.loc[x, y]
    Rxyz = math.sqrt((abs(xz ** 2) + abs(yz ** 2) - 2 * xz * yz * xy) /
                     (1 - abs(xy ** 2)))
    R2 = Rxyz ** 2
    # Calculate adjusted R-squared
    n = len(filter_data)  # Number of rows
    k = 2  # Number of independent variables
    R2_adj = 1 - (((1 - R2) * (n - 1)) / (n - k - 1))
    print("correlation for dependent level of Stress Level")
    print("R-Squared:", R2)
    print("Adjusted R-Squared:", R2_adj)


def line_chart_heart_rate_sleep_quality_ml(df):
    """
    Generates a line chart showing the trends of heart rate and
    sleep quality over time, using the provided DataFrame. The
    parameters are df: DataFrame containing sleep data with 'Start',
    'End', 'Heart rate', and 'Sleep quality' columns and returns None.
    Missing heart rate values are predicted using a decision tree
    regressor trained on the sleep quality feature.
    """

    # Convert date columns to datetime
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])

    # Remove percentage symbol from Sleep Quality column and convert to float
    df['Sleep quality'] = df['Sleep quality'].str.rstrip('%').astype('float')

    # Create subplots
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))

    # Subplot 1: Line plot for Heart Rate
    axs[0].plot(df['Start'], df['Heart rate'])
    axs[0].set_title('Heart Rate')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Heart Rate')

    # Subplot 2: Line plot for Sleep Quality
    axs[1].plot(df['Start'], df['Sleep quality'])
    axs[1].set_title('Sleep Quality')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Sleep Quality')

    # Subplot 3: Joint Line plot for Heart Rate and Sleep Quality (without ML)
    axs[2].plot(df['Start'], df['Heart rate'], label='Heart Rate')
    axs[2].plot(df['Start'], df['Sleep quality'], label='Sleep Quality')
    axs[2].set_title('Heart Rate and Sleep Quality (without ML)')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Value')
    axs[2].legend()

    # Subplot 4: Joint Line plot for Heart Rate and Sleep Quality (with ML)
    # Split the dataset into two parts: one with non-null values and
    # the other with null values
    df_with_values = df.dropna(subset=['Heart rate', 'Sleep quality'])
    df_missing_values = df[df['Heart rate'].isnull()]

    # Train a decision tree regressor to predict "Heart rate"
    X_train = df_with_values[['Sleep quality']]
    y_train = df_with_values['Heart rate']
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Use the trained model to predict missing values
    X_missing = df_missing_values[['Sleep quality']]
    y_missing_predicted = model.predict(X_missing)

    # Replace missing values in the original DataFrame
    df.loc[df['Heart rate'].isnull(), 'Heart rate'] = y_missing_predicted

    # Plot the joint Line plot with replaced values
    axs[3].plot(df['Start'], df['Heart rate'], label='Heart Rate')
    axs[3].plot(df['Start'], df['Sleep quality'], label='Sleep Quality')
    axs[3].set_title('Heart Rate and Sleep Quality (with ML)')
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('Value')
    axs[3].legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the file in jpg format
    plt.savefig("line_chart_heart_sleep_ML.jpg")

    # Display the plot
    plt.show()


def main():
    data = pd.read_csv('Sleep_Efficiency.csv')
    bar_chart_sleep_data(data)

    data2 = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    sleep_stresslevel_scatterplot(data2)

    df = pd.read_csv('sleepdata.csv', delimiter=';', skip_blank_lines=True)
    line_chart_heart_rate_sleep_quality_ml(df)


if __name__ == '__main__':
    main()
