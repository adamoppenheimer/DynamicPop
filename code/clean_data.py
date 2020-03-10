import os
import numpy as np
import pickle

cur_path = '/Users/adamalexanderoppenheimer/Desktop/DynamicPop'
os.chdir(cur_path + '/code')

import util

os.chdir(cur_path)

datadir = 'data/demographic/raw/'
newdatadir = 'data/demographic/clean/'
fert_dir = datadir + 'jpn_fertility.csv'
mort_dir = datadir + 'jpn_mortality.csv'
pop_dir = datadir + 'jpn_population.csv'

fert_data = util.get_fert_data(fert_dir)
mort_data, pop_data = util.get_mort_pop_data(mort_dir, pop_dir)

infant_mort = mort_data.iloc[0]
non_infant_mort = mort_data.iloc[1:]
infant_pop = pop_data.iloc[0].drop(2017)

pickle.dump(pop_data, open(newdatadir + 'pop.p', 'wb') )
pickle.dump( (pop_data, fert_data), open(newdatadir + 'fert.p', 'wb') )
pickle.dump( (pop_data, infant_pop, non_infant_mort, infant_mort), open(newdatadir + 'mort.p', 'wb') )

imm = util.calc_imm_resid(fert_data, mort_data, pop_data)
imm_rate = imm / pop_data
imm_rate = imm_rate.drop([1947, 2016, 2017], axis=1)

drop_immyears = np.linspace(1948, 1996, 1996 - 1948 + 1)
imm_rate = imm_rate.drop(drop_immyears, axis=1)

pickle.dump(imm_rate, open(newdatadir + 'imm.p', 'wb') )

pickle.dump((pop_data, fert_data, mort_data, imm_rate), open(newdatadir + 'all.p', 'wb') )
