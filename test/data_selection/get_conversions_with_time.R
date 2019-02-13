# Include ADNIMERGE
library(ADNIMERGE)
library(dplyr)

mci_data <- adnimerge %>%
  filter(Month == 0 & DX == "MCI")
  

mci_ids <- mci_data %>% 
  pull(PTID)
  #head(50)

# Vector for conversions
mci_processed = data_frame()

# Iterate over MCI IDs and find conversions
for (sid in mci_ids) {
  # Get the first month where Dementia was reported
  subject_data  <- adnimerge %>% filter(PTID == sid & DX == "Dementia") %>% arrange(Month) %>% head(1)
  if (nrow(subject_data)){
    # print(subject_data %>% select(PTID, DX, Month))
    conversion_month <- subject_data$Month
    subject_data <- mci_data %>% 
      filter(PTID == sid) %>%
      mutate(Month.SC = conversion_month, Month.CONVERSION = conversion_month, label = "MCIc")
    
    # Append subject and label
    mci_processed <- merge(mci_processed, subject_data, all = TRUE)
  }
}

# Now, go for stable and get time of stability reported
for (sid in mci_ids) {
  if (!sid %in% mci_processed$PTID){
    subject_data <- adnimerge %>% filter(PTID == sid & DX == "MCI") %>% arrange(desc(Month)) %>% head(1)
    stability_months <- subject_data$Month 
    subject_data <- mci_data %>% 
      filter(PTID == sid) %>%
      mutate(Month.SC = stability_months, Month.STABILITY = stability_months, label = "MCIs")
    
    # Append subject and label
    mci_processed <- merge(mci_processed, subject_data, all = TRUE)
  }
}

# Save results to file
#write.csv(mci_processed, './param/df_conversions_with_times.csv')

# Visualize results
#mci_processed$Month.CONVERSION = as.numeric(mci_processed$Month.CONVERSION)
#mci_processed$Month.STABILITY = as.numeric(mci_processed$Month.STABILITY)

for (month in c(12, 24, 48)) {
  summ <- mci_processed %>%
    filter(
      as.numeric(as.character(Month.STABILITY)) > month | 
      as.numeric(as.character(Month.CONVERSION)) == month)%>%
    group_by(label) %>%
    summarise(Total=n(), MeanAge=mean(AGE, na.rm = TRUE), StdAge=sd(AGE, na.rm = TRUE))
  print(sprintf("At least %d months of stability and %d months for conversion", month, month))
  print(summ)
}

library(ggplot2)
mci_processed %>%
  filter(!Month.SC %in% c(0, 102, 114, 126, 150)) %>%
  mutate(Month.SC = as.numeric(as.character(Month.SC))) %>%
  group_by(label) %>%
  ggplot(aes(x = label, fill = label)) + geom_bar() + facet_wrap(~Month.SC)
