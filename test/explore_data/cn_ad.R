library(ADNIMERGE)
library(dplyr)

# Get HC and AD
hc_ad <- adnimerge %>%
  filter(DX.bl %in% c('CN', 'AD') & DX %in% c('CN', 'Dementia') & Month == 0 & VISCODE == 'bl' & COLPROT == 'ADNI1') %>%
  select(PTID, DX, PTGENDER)

# Print some info
hc_ad %>%
  group_by(DX, PTGENDER) %>%
  summarize(TOTAL = n())