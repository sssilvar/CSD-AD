# Include ADNIMERGE
library(ADNIMERGE)

data_bl_mci = subset(adnimerge,Month==0 & DX=='MCI')
data_mci = subset(adnimerge,RID %in% unique(data_bl_mci$RID))

# Extract Conversions
sids = vector()

for (sub in subset(data_mci, Month==0)$PTID) {
  dxs_mci = subset(adnimerge,PTID==sub)$DX
  last_dx = dxs_mci[which(!is.na(dxs_mci))][length(dxs_mci[which(!is.na(dxs_mci))])]
  if (last_dx%in%c('Dementia','MCI to Dementia')){
    
    # Append PTID
    sids = append(sids, sub)
  }
}

conversions = data_bl_mci[ (data_bl_mci$PTID %in% sids), ]
stable = data_bl_mci[ !(data_bl_mci$PTID %in% sids), ]