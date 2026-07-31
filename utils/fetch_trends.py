from pytrends.request import TrendReq

def get_trend_score(place, geo='IN'):
    try:
        pytrends = TrendReq(hl='en-US', tz=330)
        pytrends.build_payload([place], cat =0, timeframe='now 7-d', geo=geo)
        data = pytrends.interest_over_time()
        
        if not data.empty:
            return int(data[place].mean())
        else:
            return 0
    except Exception as e:
        return 0
