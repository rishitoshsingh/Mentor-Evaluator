from datetime import datetime
import re
from collections import OrderedDict
import numpy as np
import utils

start_time = '11:17:39'
start_time = datetime.strptime(start_time, '%H:%M:%S')
transcript = OrderedDict()

with open('transcripts/apple.txt', 'r') as file:
    for line in file:
        time_match = re.search('[0-9]{2}:[0-9]{2}:[0-9]{2}', line)
        current_time = datetime.strptime(line[time_match.start():time_match.end()], '%H:%M:%S')
        delta = current_time - start_time
        delta = utils.get_time(delta.total_seconds())
        transcript[delta] = line[time_match.end()+1:].strip()

import pandas as pd
data = pd.read_excel('timestamps-apple.xlsx')
data = data.to_dict('list')
del data['slide']
data['cc'] = ['']*len(data['slide_no'])

def compare_time(t1, t2):
    if t1 is np.nan or t2 is np.nan:
        return False
    t1 = datetime.strptime(t1,"%H:%M:%S")
    t2 = datetime.strptime(t2,"%H:%M:%S")
    return t1 >= t2

current_slide = 0
end_triggered = False
for cc_time in transcript.keys():
    if compare_time(data['end_time'][current_slide], cc_time):
        data['cc'][current_slide] = data['cc'][current_slide] + transcript[cc_time] + ' '
    elif end_triggered:
        data['cc'][current_slide] = data['cc'][current_slide] + transcript[cc_time] + ' '
    else:
        while True:
            current_slide += 1
            if current_slide == len(data['end_time'])-1:
                end_triggered = True
                break
            elif data['end_time'][current_slide] is not np.nan and compare_time(data['end_time'][current_slide], cc_time):
                break
        data['cc'][current_slide] = data['cc'][current_slide] + ' ' + transcript[cc_time]

import pandas as pd
data = pd.DataFrame(data)
data.to_csv('temp.csv')
