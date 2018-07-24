# Load libraries
library(ADNIMERGE)

workdir <- '/user/ssilvari/home/Documents/structural/CSD-AD/param/common'
merge_csv <- file.path(workdir, 'adnimerge.csv')
mmse_csv <- file.path(workdir,  'mmse.csv')
cdr_csv <- file.path(workdir,  'cdr.csv')

# Load ADNIMERGE
write.csv(adnimerge, file = merge_csv)
write.csv(mmse, file = mmse_csv)
write.csv(cdr, file = cdr_csv)
