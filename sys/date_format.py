from datetime import datetime, timezone
import numpy as np



"""
string to datetime
"""
str2format_time = {
	"normal" : lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(tzinfo=None),
	"upbit": lambda x: datetime.strptime(x.replace("T", " "), '%Y-%m-%d %H:%M:%S').replace(tzinfo=None),
	"empty": lambda x: x,
}


"""
datetime to string
"""
format_time2str = {
	"normal" : lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
}


"""
UTC(datetime) to timestamp
"""
UTC_to_timestamp = {
	"datetime": lambda x: x.replace(tzinfo=timezone.utc).timestamp()
}

"""
timestamp to UTC
"""
timestamp_to_UTC = {
	"normal": lambda x: datetime.utcfromtimestamp(x).replace(tzinfo=None),
	"ms": lambda x: datetime.utcfromtimestamp(x/1000).replace(tzinfo=None)
}


	

def convert_datetime64_to_datetime(dt64):	
	"""
	numpy 의 datetime64 를 datetime 로 convert
	"""	
	ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
	return datetime.utcfromtimestamp(ts)

	
