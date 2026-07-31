import matplotlib.pyplot as plt
def plot_historical_patterns(df, site, trip_date, years=None):
    fig, ax = plt.subplots()
    
    if not years:
        years = [trip_date.year - 1 - i for i in range(3)]
        
    for year in years:
        mask = (
            (df['site'] == site) &
            (df['date'].dt.month == trip_date.month) &
            (df['date'].dt.day.between(trip_date.day-7, trip_date.day+7)) &
            (df['date'].dt.year == year)
        )
        hist = df[mask]
        if not hist.empty:
            ax.plot(
                hist['date'].dt.day,
                hist['tourists'].values,
                label=str(year)
            )
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Tourist Count")
    ax.set_title(f"Historical Tourists ±7 Days around {trip_date.strftime('%b %d')}")
    ax.legend()
    return fig
