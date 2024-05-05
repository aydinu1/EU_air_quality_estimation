import plotly.express as px

def plot_air_quality_map(df, plot_prediction=False):

    if plot_prediction:
        if "pm2_5" in df.columns:
            df = df.drop(["pm2_5"])
        df = df.rename(columns={"predicted_pm25": "pm2_5"})

    df["time"] = df.index
    # Plot the map
    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        text='city',
        size='pm2_5',
        color='pm2_5',
        hover_name='city',
        projection='natural earth',
        size_max=30,
        hover_data='time'
    )

    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="DarkBlue",
        showland=True,
        landcolor="LightGrey",
        showocean=True,
        oceancolor="LightBlue",
        showlakes=True,
        lakecolor="LightBlue",
        center=dict(lat=53, lon=9),  # Center of Europe (approx.)
        scope='europe',              # Show Europe map
        projection_rotation=dict(lon=10)  # Rotate the map slightly for better view
    )

    # fig.update_layout(title_text='Fine Particle Pollution in EU Capitals', title_x=0.5)
    # Set the figure size
    fig.update_layout(
        title_text='Predictions for the Next Hour (GMT+0)', title_x=0.5,
        geo=dict(showframe=True, showcoastlines=True),
        width=800,  # Adjust the width as needed
        height=600,  # Adjust the height as needed
    )

    fig.update_traces(
        textfont=dict(color="black")
    )

    # fig.show()
    return fig
