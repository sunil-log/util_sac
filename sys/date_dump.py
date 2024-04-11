



# -------------------------------
# 주어진 datatime range 를 12hr 로 끊어서,
# 가까운 datetime 끼리 pair 를 만들어 return 한다.
# return 값은 string 이다
# 
# 그런데 pandas 안의 sampling method 를 사용하면서 사용 안하게 됌
# -------------------------------
def split_datetime_range_12hr(start_datetime, end_datetime):
	
	time_format = "%Y-%m-%d %H:%M:%S"

	# convert to datetime
	start_datetime = pd.to_datetime(start_datetime)
	end_datetime = pd.to_datetime(end_datetime)

	# if the interval is less than 12hr, return the two datetime
	if (end_datetime - start_datetime) < pd.Timedelta(hours=12):
		# convert to string
		start_datetime = start_datetime.strftime(time_format)
		end_datetime = end_datetime.strftime(time_format)
		# return
		return np.array([[start_datetime, end_datetime]])

	# get datetime range
	datetime_range = pd.date_range(start_datetime, end_datetime, freq="12H")

	# convert to dataframe
	df = pd.DataFrame(datetime_range, columns=["date"])

	# new column which is shifting date column
	#   and fill last na by end_datetime
	#   위 freq=12H 가 12H로 나눈 나머지는 버리길래
	df["date_shift"] = df["date"].shift(-1)
	df["date_shift"] = df["date_shift"].fillna(end_datetime)

	# date_shift-date > 1min 보다 큰 경우만 추출
	df = df[df["date_shift"] - df["date"] > pd.Timedelta(minutes=1)]

	# convert all column to string
	df = df.astype(str)

	# convert to numpy array
	datetime_range_12h = df.to_numpy()

	return datetime_range_12h
	



# ==================================
# main
# 	Input
# 		[t1, t1, t1, .., t2, t2, t2, ..., t9, t9, t9]
# 	Output
# 		{t1: t2, t2: t3, ..., t9: t_now}
# ==================================
def time_range_dict(s_time):	

	s_time = s_time.drop_duplicates()
	s_time = s_time.sort_values()

	df_time_range = pd.DataFrame({"t_start": s_time})
	df_time_range['t_end'] = df_time_range['t_start'].shift(-1)
	
	t_now = format_time2str['normal'](datetime.now())
	df_time_range = df_time_range.fillna(t_now) 	# fill Na (the last t_end) by t_now
	d_next_time = dict( zip(df_time_range['t_start'], df_time_range['t_end']) )
	
	return d_next_time
	
