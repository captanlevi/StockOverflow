import datetime

MAX_HOLDING = 5
MAX_TRADE_TIME_IN_DAYS = 3
ROLLING_WINDOW_SIZE_IN_DAYS = 10

# for minute 9:15 to 3:29
# for 5 minute 9:15 to 3:25
start = datetime.datetime(year= 2024, month= 1, day= 1, hour= 9, minute= 15)
start_time = datetime.time(hour= 9, minute= 15)
end_minute =  datetime.datetime(year= 2024, month= 1, day= 1, hour= 15, minute= 29)
end_time_minute = datetime.time(hour= 15, minute= 29)
end_5_minute = datetime.datetime(year= 2024, month= 1, day= 1, hour= 15, minute= 25)
end_time_5_minute = datetime.time(hour= 15, minute= 25)
minutes_in_day = int(((end_minute - start).total_seconds()/60)) + 1  # +1 cause the first one also counts
minutes_5_in_day = int(((end_5_minute - start).total_seconds()/(60*5))) + 1
ROLLING_WINDOW_SIZE_IN_MINUTE = int(minutes_in_day*ROLLING_WINDOW_SIZE_IN_DAYS)
ROLLING_WINDOW_SIZE_IN_5_MINUTE = int(minutes_5_in_day*ROLLING_WINDOW_SIZE_IN_DAYS)

MAX_TRADE_TIME_IN_MINUTE = int(minutes_in_day*MAX_TRADE_TIME_IN_DAYS)
MAX_TRADE_TIME_IN_5_MINUTE = int(minutes_5_in_day*MAX_TRADE_TIME_IN_DAYS)

MODES = ["minute","5minute","day"]