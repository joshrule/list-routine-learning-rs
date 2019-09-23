library(tidyverse)
library(stringr)
library(tidyboot)

import_data <- function() {
  read_csv("computational_tests/raw_data.csv")
}

plot_data <- function(df) {
  df %>% group_by(rule) %>% group_walk(plot_single_rule)
  NULL
}

plot_single_rule <- function(df, args) {

  rule <- str_replace_all(args[[1,"rule"]], " ", "")

  key_hs <- make_key_hs(df)
  other_hs <- make_other_hs(df)

  ggplot() +

    geom_line(data=other_hs, aes(x=n_data, y=mean, group=label), color = "#aaaaaa", alpha=1, size=0.2) +

    geom_errorbar(data=key_hs, aes(x=n_data, ymin=ci_lower, ymax=ci_upper, group=label), color="#666666", width=0.1) +
    geom_line(data=key_hs, aes(x=n_data, y=empirical_stat, color=label, group=label), size = 1.0) +

    xlab("# of Data") +
    ylab("Normalized Log Posterior") +
    ggsave(paste0("computational_tests/", rule, ".png"))
}

make_key_hs <- function(df) {
  df %>%
    filter(label %in% c("correct", "memorized", "empty")) %>%
    group_by(n_data, label) %>%
    tidyboot_mean(col = norm_posterior)
}

make_other_hs <- function(df) {
  labels <- df %>% filter(log_posterior != -Inf & !(label %in% c("correct", "memorized", "empty"))) %>% select(label) %>% unique()
  df %>%
    filter(label %in% labels$label) %>%
    group_by(n_data, label) %>%
    dplyr::summarize(mean = mean(norm_posterior))
}
