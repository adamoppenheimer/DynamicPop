
rm(list=ls())

library(rlang)
library(reticulate)
use_python('/usr/local/bin/python')

library(demography)

getwd()
setwd('/Users/adamalexanderoppenheimer/Desktop/DynamicPop/data/demographic')

pd <- import('pandas')
fert_full <- pd$read_pickle('clean/fert.p')
pop_data <- fert_full[[1]]
fert_data <- fert_full[[2]]
mort_full <- pd$read_pickle('clean/mort.p')
non_inf_mort <- mort_full[[3]]
inf_mort <- mort_full[[4]]
mort_data = matrix(rbinom(100 * 70, 1, 0.5), ncol = 70, nrow = 100)
for (i in 1:dim(inf_mort)) {
  mort_data[,i] = prepend(unlist(non_inf_mort[i]), unlist(inf_mort[i]))
}
mort_data = as.data.frame(mort_data)
colnames(mort_data) <- colnames(non_inf_mort)
imm_data <- pd$read_pickle('clean/imm.p')

########################################
##### Construct Demography Objects #####
########################################
# Box-Cox (lambda) values: 0 (for mortality), 0.4 (for fertility) and 1 (for migration)
# Source: https://www.rdocumentation.org/packages/demography/versions/1.22/topics/read.demogdata
demog_obj_fert = demogdata(fert_data, pop_data[15:51, 1:68], as.numeric(rownames(fert_data)), as.numeric(colnames(pop_data[,1:68])), 'fertility', 'Japan', 'Total', 0.4)
demog_obj_mort = demogdata(mort_data, pop_data[, 1:70], as.numeric(rownames(imm_data)), as.numeric(colnames(pop_data[, 1:70])), 'mortality', 'Japan', 'Total', 0)
demog_obj_imm = demogdata(imm_data, pop_data[51:69], as.numeric(rownames(imm_data)), as.numeric(colnames(pop_data[51:69])), 'migration', 'Japan', 'Total', 1)

# Source: https://www.rdocumentation.org/packages/demography/versions/1.22/topics/fdm
# Set transform = FALSE to ensure past data is able to be used in graphs
fdm_fert = fdm(demog_obj_fert, transform = FALSE)
fdm_mort = fdm(demog_obj_mort, transform = FALSE)
fdm_imm = fdm(demog_obj_imm, transform = FALSE)

# Forecast variables up to 2100 (up to 2050 for model, up to 3000 for graphs)
forecast_fert = forecast(fdm_fert, h=986)
forecast_mort = forecast(fdm_mort, h=984)
forecast_imm = forecast(fdm_imm, h=985)

fitted_fert = forecast_fert['fitted'][[1]]['y']
fitted_mort = forecast_mort['fitted'][[1]]['y']
fitted_imm = forecast_imm['fitted'][[1]]['y']

write.csv(fitted_fert,'r_forecasts/fert_fitted.csv', row.names = FALSE)
write.csv(fitted_mort,'r_forecasts/mort_fitted.csv', row.names = FALSE)
write.csv(fitted_imm,'r_forecasts/imm_fitted.csv', row.names = FALSE)

future_fert = forecast_fert['rate']
future_mort = forecast_mort['rate']
future_imm = forecast_imm['rate']

write.csv(future_fert,'r_forecasts/fert_pred.csv', row.names = FALSE)
write.csv(future_mort,'r_forecasts/mort_pred.csv', row.names = FALSE)
write.csv(future_imm,'r_forecasts/imm_pred.csv', row.names = FALSE)












