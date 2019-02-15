library(ADNIMERGE)
library(dplyr)
library(zoo)
library(ggplot2)

# Get baseline subjects
baseline <- adnimerge %>% filter(DX.bl %in% c("EMCI", "LMCI", "MCI") & Month == 0 & DX == "MCI") #%>% head(100)

# Dataframe where MCI data will be saved
mci_data <- data_frame()

for (sid in baseline$RID) {
  subject_data <- adnimerge %>% filter(RID == sid)
  if ("Dementia" %in% subject_data$DX) {
    dxs <- na.locf(subject_data %>% arrange(Month) %>% pull(DX), fromLast = TRUE)
    dx_change <- diff(as.numeric(dxs))
    
    # Detect regression to MCI
    if (!-1 %in% dx_change) {
      month <- subject_data %>% filter(DX == "Dementia") %>% arrange(Month) %>% head(1) %>% pull(Month)
      subject_valid <- subject_data %>%
        filter(Month == 0) %>%
        mutate(target = "MCIc", Month.SC = month, Month.CONVERSION = month)
      mci_data <- merge(mci_data, subject_valid, all = TRUE)
    }
  } else {
    dxs <- na.locf(subject_data %>% arrange(Month) %>% pull(DX), fromLast = TRUE)
    dx_change <- diff(as.numeric(dxs))
    
    # Detect regression to control
    if (!-1 %in% dx_change) {
      month <- subject_data %>% arrange(desc(Month)) %>% head(1) %>% pull(Month)
      subject_valid <- subject_data %>%
        filter(Month == 0) %>%
        mutate(target = "MCInc", Month.SC = month, Month.STABLE = month)
      mci_data <- merge(mci_data, subject_valid, all = TRUE)
    }
  }
}

mci_non_zero <- mci_data %>% filter(!Month.SC == 0)

for (month in c(24, 36, 60)) {
  summ <- mci_non_zero %>%
    filter(
      as.numeric(as.character(Month.STABLE)) >= month | 
        as.numeric(as.character(Month.CONVERSION)) <= month)%>%
    group_by(target) %>%
    summarise(Total=n(), MeanAge=mean(AGE, na.rm = TRUE), StdAge=sd(AGE, na.rm = TRUE)) %>%
    arrange(desc(target))
  print(sprintf("At least %d months of stability and %d months for conversion", month, month))
  print(summ)
}

write.csv(mci_non_zero, "./param/df_conversions_with_times.csv")
