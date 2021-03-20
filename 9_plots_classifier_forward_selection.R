library(ggplot2)

recall <- c(0.2396622015142691, 0.32201514269073966, 0.361211415259173, 0.37705299941758885, 0.3829353523587653,
            0.3829353523587653, 0.38887594641817125, 0.38887594641817125, 0.38887594641817125, 0.38887594641817125,
            0.38887594641817125, 0.38887594641817125, 0.38887594641817125, 0.38887594641817125, 0.38887594641817125,
            0.38887594641817125, 0.38887594641817125, 0.38887594641817125, 0.38887594641817125, 0.3829353523587653)

accuracy <- c(0.8397890347111492, 0.8409275546352479, 0.8409275546352479, 0.8409275546352479, 0.8409275546352479, 
              0.8409275546352479, 0.8409275546352479, 0.8409275546352479, 0.8409275546352479, 0.8409275546352479,
              0.8409275546352479, 0.8409275546352479, 0.8409275546352479, 0.8409275546352479, 0.8409275546352479,
              0.8409275546352479, 0.8409275546352479, 0.8409275546352479, 0.8409275546352479, 0.8409275546352479)

# find optimal step
sOpt <- which.max(recall) # step count starts with 0

dfRecall <- data.frame(Step = 1:20, Recall = recall)
ggplot() +
  geom_vline(aes(xintercept = sOpt), color = "darkgrey") +
  geom_line(aes(x = Step, y = Recall), dfRecall) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())


# find optimal step
sOpt <- which.max(accuracy)  # step count starts with 0

dfAcc <- data.frame(Step = 1:20, Accuracy = accuracy)
ggplot() +
  geom_vline(aes(xintercept = sOpt), color = "darkgrey") +
  geom_line(aes(x = Step, y = Accuracy), dfAcc) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
